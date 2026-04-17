"""
Task 12: MOP Routing Clustering + Automatic Cluster Prompt Generation

Phase A — K-Means clustering with Silhouette Score-based K selection.
Phase B — Gemini analyzes each cluster's centroid features and generates
          a bespoke MOP prompt template per cluster, saved to
          data/cache/mop_prompts.json for downstream use by pipeline_5.

Prompts can be reviewed and manually refined after inspecting the 10K
cluster distribution. Re-running this script regenerates them automatically.
"""
import os
import sys
import json
import asyncio
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

client = genai.Client()

MOP_PROMPTS_PATH = "data/cache/mop_prompts.json"
K_REPORT_PATH = "data/eval_results/k_selection_report.json"

SCENE_LABELS = [
    "standard continuous grocery aisle shelf",
    "promotional endcap at the end of an aisle",
    "grocery checkout area or till",
    "standalone promotional display bin",
]


def extract_features(item: dict) -> list:
    """
    Extract 5 numerical features per image for clustering.
    Features encode the visual and operational characteristics of each image.
    """
    f1_scene_conf = float(item["L1_scene"]["confidence"])
    f2_num_fixtures = float(item["L2_fixtures"]["num_fixtures"])

    f3_tidy_conf = 0.5
    try:
        f3_tidy_conf = float(item["L4_attributes"]["operational_state"]["tidiness"]["confidence"])
        if "neatly organized" not in item["L4_attributes"]["operational_state"]["tidiness"]["label"]:
            f3_tidy_conf = 1.0 - f3_tidy_conf
    except KeyError:
        pass

    f4_stock_conf = 0.5
    try:
        f4_stock_conf = float(item["L4_attributes"]["operational_state"]["stock_level"]["confidence"])
    except KeyError:
        pass

    f5_promo = 0.0
    try:
        promo_label = item["L4_attributes"]["operational_state"]["promotion"]["label"]
        if "promotional" in promo_label.lower():
            f5_promo = 1.0
    except KeyError:
        pass

    return [f1_scene_conf, f2_num_fixtures, f3_tidy_conf, f4_stock_conf, f5_promo]


def select_optimal_k(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 8) -> tuple:
    """
    Evaluate K-Means for each K using Silhouette Score and return the best K.
    Silhouette Score is preferred over Elbow Method: it yields a single,
    unambiguous numerical value per K, making it defensible in academic writing.
    """
    scores = {}
    k_max = min(k_max, len(X_scaled) - 1)

    print("\n[K-SELECTION] Silhouette Score analysis:")
    print(f"{'K':>4} | {'Silhouette Score':>18}")
    print("-" * 28)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = round(float(score), 4)
        print(f"{k:>4} | {score:>18.4f}")

    optimal_k = max(scores, key=scores.get)
    print(f"\n[K-SELECTION] Optimal K = {optimal_k} (Silhouette Score = {scores[optimal_k]})")
    return optimal_k, scores


def describe_cluster_centroid(centroid: list, cluster_id: int, labels: np.ndarray, data: list) -> dict:
    """
    Build a human-readable summary of a cluster's dominant characteristics
    by examining the centroid values and representative samples.
    """
    f1_scene, f2_fixtures, f3_tidy, f4_stock, f5_promo = centroid

    # Collect dominant scene label from items in this cluster
    scene_labels_in_cluster = []
    for i, label in enumerate(labels):
        if label == cluster_id:
            scene = data[i]["L1_scene"].get("predicted_scene", "unknown scene")
            scene_labels_in_cluster.append(scene)

    from collections import Counter
    dominant_scene = Counter(scene_labels_in_cluster).most_common(1)[0][0] if scene_labels_in_cluster else "unknown"

    return {
        "cluster_id": cluster_id,
        "n_samples": int(np.sum(labels == cluster_id)),
        "dominant_scene": dominant_scene,
        "avg_scene_confidence": round(float(f1_scene), 3),
        "avg_fixtures_count": round(float(f2_fixtures), 2),
        "avg_tidiness_score": round(float(f3_tidy), 3),
        "avg_stock_confidence": round(float(f4_stock), 3),
        "promotional_majority": bool(f5_promo > 0.5),
    }


async def generate_cluster_prompt(cluster_desc: dict) -> str:
    """
    Ask Gemini to craft a MOP prompt template tailored to this cluster's
    statistical characteristics. The prompt instructs the downstream VLM
    on how to caption images belonging to this routing group.
    """
    prompt = f"""You are an expert in designing prompts for Vision-Language Models (VLMs)
used in grocery retail image captioning.

Based on the following cluster profile, write a SYSTEM PROMPT TEMPLATE for a retail VLM.

The prompt must follow an OPEN+ANCHOR strategy:
  - The VLM is free to describe WHAT IT SEES (visual layout, colours, seasonal themes, arrangement).
  - It MUST incorporate specific verified product anchors (provided via {{l3_str}}).
  - Ambiguous operational attributes (provided via {{ambiguous}}) must be explicitly withheld
    (the VLM should state they cannot be confirmed rather than guessing).
  - Observable confirmed operational details (provided via {{obs_str}}) may be included naturally.

The prompt must:
1. Begin with a retail analyst persona relevant to the scene type.
2. Instruct the VLM to write 2-3 natural sentences.
3. Tell the VLM to describe what it visually observes first.
4. Include a VERIFIED ANCHORS block using the placeholder {{l3_str}} that the VLM MUST mention.
5. Use {{ambiguous}} to list attributes the VLM must NOT guess.
6. Include {{obs_str}} for confirmed observable operational details.
7. Use {{ctx_str}} as background scene context only (not a product fact source).
8. Include {{l1}} (scene type) and {{l2_str}} (fixtures) for spatial framing.

IMPORTANT: Do NOT instruct the VLM to "only use the provided facts" or restrict it to tags only.
The VLM should use its visual understanding freely, anchored by the verified facts.

Cluster Profile:
- Cluster ID: {cluster_desc['cluster_id']}
- Dominant scene type: {cluster_desc['dominant_scene']}
- Average fixture count: {cluster_desc['avg_fixtures_count']}
- Average tidiness score: {cluster_desc['avg_tidiness_score']} (higher = neater)
- Average stock confidence: {cluster_desc['avg_stock_confidence']}
- Promotional scene majority: {cluster_desc['promotional_majority']}
- Sample count: {cluster_desc['n_samples']}

Return ONLY the raw prompt string. Do not wrap in markdown or add extra explanation."""

    # Only use gemini-2.5-flash — retry with exponential backoff on 503 overload
    model_name = "gemini-2.5-flash"
    max_retries = 5

    for attempt in range(1, max_retries + 1):
        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.4),
            )
            if not response.text:
                raise Exception(f"Empty Gemini response for cluster {cluster_desc['cluster_id']}")
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            is_503 = "503" in err_str or "UNAVAILABLE" in err_str
            is_last_attempt = attempt == max_retries

            if is_503 and not is_last_attempt:
                wait_s = 2 ** attempt  # 2, 4, 8, 16, 32 seconds
                print(f"  [RETRY] Cluster {cluster_desc['cluster_id']} | 503 overloaded. "
                      f"Retrying in {wait_s}s (attempt {attempt}/{max_retries})...")
                await asyncio.sleep(wait_s)
            else:
                raise  # non-503 error, or all retries exhausted — propagate


async def generate_all_prompts(cluster_descriptions: list) -> dict:
    """Concurrently generate a MOP prompt for each cluster."""
    tasks = [generate_cluster_prompt(desc) for desc in cluster_descriptions]
    prompts_list = await asyncio.gather(*tasks)

    prompts = {}
    for desc, prompt_text in zip(cluster_descriptions, prompts_list):
        cid = desc["cluster_id"]
        prompts[cid] = {
            "cluster_profile": desc,
            "mop_prompt_template": prompt_text,
            "note": (
                "Auto-generated by Gemini based on cluster centroid features. "
                "Review and refine manually after inspecting cluster distribution."
            )
        }
        print(f"\n[PROMPT GEN] Cluster {cid} prompt generated ({len(prompt_text)} chars).")

    return prompts


def run_clustering():
    input_path = "data/cache/hierarchical_tags_final.json"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Run L4 tagging pipeline first.")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 samples to cluster.")

    print(f"Loaded {n} image tag records for clustering.")

    X_raw = [extract_features(item) for item in data]
    X = np.array(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Phase A: K selection via Silhouette Score ─────────────────────────────
    k_max = 8 if n >= 20 else 2
    optimal_k, silhouette_scores = select_optimal_k(X_scaled, k_min=2, k_max=k_max)

    # ── Final clustering with optimal K ───────────────────────────────────────
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Assign clusters and save
    routes = []
    cluster_counts = {}
    for i, item in enumerate(data):
        cluster_id = int(labels[i])
        item["MOP_route_cluster"] = cluster_id
        routes.append(item)
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        print(f"[{item['image_file']}] -> Routing Cluster {cluster_id}")

    print("\n[CLUSTER DISTRIBUTION]")
    for cid, count in sorted(cluster_counts.items()):
        print(f"  Cluster {cid}: {count} images ({100*count/n:.1f}%)")

    out_path = "data/cache/clustered_routes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(routes, f, indent=4)

    os.makedirs("data/eval_results", exist_ok=True)
    with open(K_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "n_samples": n,
            "optimal_k": optimal_k,
            "silhouette_scores_by_k": silhouette_scores,
            "cluster_distribution": cluster_counts,
            "note": (
                "Silhouette Score used for K selection (preferred over Elbow Method "
                "for objective, single-value academic justification). "
                "Higher score indicates better-separated MOP routing clusters."
            )
        }, f, indent=4)

    # ── Phase B: Auto-generate MOP prompts per cluster via Gemini ─────────────
    print(f"\n[PROMPT GEN] Generating {optimal_k} cluster-specific MOP prompts via Gemini...")
    cluster_descriptions = [
        describe_cluster_centroid(
            centroid=kmeans.cluster_centers_[cid].tolist(),
            cluster_id=cid,
            labels=labels,
            data=data,
        )
        for cid in range(optimal_k)
    ]

    # Run async prompt generation and explicitly clean up the event loop
    # to prevent aiohttp connector __del__ errors causing non-zero exit codes.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        mop_prompts = loop.run_until_complete(generate_all_prompts(cluster_descriptions))
    finally:
        # Cancel any lingering tasks before closing
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    os.makedirs("data/cache", exist_ok=True)
    with open(MOP_PROMPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(mop_prompts, f, indent=4, ensure_ascii=False)

    print(f"\nSaved clustered routes        -> {out_path}")
    print(f"Saved K-selection report      -> {K_REPORT_PATH}")
    print(f"Saved auto-generated prompts  -> {MOP_PROMPTS_PATH}")
    print("MOP Routing completed successfully.")


if __name__ == "__main__":
    run_clustering()
    sys.exit(0)

"""
Task 12: MOP Routing Clustering + Automatic Cluster Prompt Generation

Phase A — K-Medoids clustering with Gower's Distance and Silhouette Score-based K selection.
Phase B — Gemini analyzes each cluster's centroid features and generates
          a bespoke MOP prompt template per cluster, saved to
          data/cache/mop_prompts.json for downstream use by pipeline_5.
"""
import os
import sys
import json
import asyncio
import time
import random
import argparse
import numpy as np
import pandas as pd
import gower
from pyclustering.cluster.kmedoids import kmedoids
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

def select_optimal_k(X_raw: list, k_min: int = 2, k_max: int = 8) -> tuple:
    random.seed(42)  # Reproducible clustering
    data_df = pd.DataFrame(X_raw, columns=['scene_conf', 'num_fixtures', 'tidy_conf', 'stock_conf', 'is_promo'])
    
    # Compute Gower's distance matrix for mixed data types
    dist_matrix = gower.gower_matrix(data_df)
    
    scores = {}
    k_max = min(k_max, len(data_df) - 1)
    
    print("\n[K-SELECTION] Silhouette Score analysis (Gower + K-Medoids):")
    print(f"{'K':>4} | {'Silhouette Score':>18}")
    print("-" * 28)
    
    # Convert numpy array to list of lists for pyclustering
    dist_matrix_list = dist_matrix.tolist()
    
    for k in range(k_min, k_max + 1):
        initial_medoids = random.sample(range(len(data_df)), k)
        kmedoids_instance = kmedoids(dist_matrix_list, initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()
        
        clusters = kmedoids_instance.get_clusters()
        
        labels = np.zeros(len(data_df))
        for cluster_id, cluster_indices in enumerate(clusters):
            for idx in cluster_indices:
                labels[idx] = cluster_id
                
        # Silhouette score supports precomputed distance metrics
        score = silhouette_score(dist_matrix, labels, metric='precomputed')
        scores[k] = round(float(score), 4)
        print(f"{k:>4} | {score:>18.4f}")

    optimal_k = max(scores, key=scores.get)
    print(f"\n[K-SELECTION] Optimal K = {optimal_k} (Silhouette Score = {scores[optimal_k]})")
    
    # Final run with optimal K
    best_initial_medoids = random.sample(range(len(data_df)), optimal_k)
    best_kmedoids = kmedoids(dist_matrix_list, best_initial_medoids, data_type='distance_matrix')
    best_kmedoids.process()
    best_clusters = best_kmedoids.get_clusters()
    
    best_labels = np.zeros(len(data_df), dtype=int)
    for cluster_id, cluster_indices in enumerate(best_clusters):
        for idx in cluster_indices:
            best_labels[idx] = cluster_id
            
    medoids_indices = best_kmedoids.get_medoids()
    centroids = [X_raw[idx] for idx in medoids_indices]
        
    return optimal_k, best_labels, centroids, scores

def describe_cluster_centroid(centroid: list, cluster_id: int, labels: np.ndarray, data: list) -> dict:
    f1_scene, f2_fixtures, f3_tidy, f4_stock, f5_promo = centroid

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
1. Begin with an objective description of the physical shelf state without assigning any subjective persona.
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
                raise  

async def generate_all_prompts(cluster_descriptions: list) -> dict:
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

def run_clustering(force_k: int = None):
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

    # ── Phase A: K selection via Silhouette Score ─────────────────────────────
    k_max = 8 if n >= 20 else 2
    
    if force_k is not None:
        print(f"\n[K-SELECTION] Bypassing automatic selection. Forcing K={force_k} as requested.")
        optimal_k, labels, centroids, silhouette_scores = select_optimal_k(X_raw, k_min=force_k, k_max=force_k)
    else:
        optimal_k, labels, centroids, silhouette_scores = select_optimal_k(X_raw, k_min=2, k_max=k_max)

    # Assign clusters and save
    routes = []
    cluster_counts = {}
    for i, item in enumerate(data):
        cluster_id = int(labels[i])
        item["MOP_route_cluster"] = cluster_id
        routes.append(item)
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        print(f"[{item.get('image_file', f'image_{i}')}] -> Routing Cluster {cluster_id}")

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
                "Gower's distance matrix was used in K-Medoids for mixed types."
            )
        }, f, indent=4)

    # ── Phase B: Auto-generate MOP prompts per cluster via Gemini ─────────────
    print(f"\n[PROMPT GEN] Generating {optimal_k} cluster-specific MOP prompts via Gemini...")
    cluster_descriptions = [
        describe_cluster_centroid(
            centroid=centroids[cid],
            cluster_id=cid,
            labels=labels,
            data=data,
        )
        for cid in range(optimal_k)
    ]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        mop_prompts = loop.run_until_complete(generate_all_prompts(cluster_descriptions))
    finally:
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
    parser = argparse.ArgumentParser(description="MOP Routing Clustering")
    parser.add_argument("--force-k", type=int, default=None,
                        help="Force a specific K value instead of using Silhouette optimal K.")
    args = parser.parse_args()
    
    run_clustering(force_k=args.force_k)
    sys.exit(0)

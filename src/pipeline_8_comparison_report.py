"""
Pipeline 8: MOP vs Vanilla Baseline Comparison Report
Generates a comprehensive side-by-side comparison between:
  - MOP-guided captions   (data/cache/final_captions.json)
  - Vanilla LLaVA captions (data/cache/baseline_captions.json)

Produces:
  1. data/eval_results/comparison_chair.json   — CHAIR scores for both models
  2. data/eval_results/comparison_report.md    — Human-readable markdown report
     with per-image caption pairs, score delta, and hallucination details
"""
import os
import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

# ── Paths ──────────────────────────────────────────────────────────────────
MOP_PATH      = "data/cache/final_captions.json"
BASELINE_PATH = "data/cache/baseline_captions.json"
LLM_JUDGE_PATH = "data/eval_results/llm_judge_scores.json"
OUT_JSON      = "data/eval_results/comparison_chair.json"
OUT_REPORT    = "data/eval_results/comparison_report.md"

lemmatizer = WordNetLemmatizer()

SAFE_WORDS = {
    "image", "photo", "variety", "types", "status", "condition", "store", "area",
    "item", "brand", "text", "visible", "product", "place", "bottle", "information",
    "retailer", "state", "sign", "shelf", "display", "section", "aisle", "report",
    "inspection", "grocery", "retail", "promotional", "signage", "region",
}


def download_nltk():
    for r in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
              "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"]:
        nltk.download(r, quiet=True)


def lemmatize(word: str) -> str:
    return lemmatizer.lemmatize(word.lower(), pos="n")


def word_boundary_match(needle: str, haystack: str) -> bool:
    pattern = re.compile(r"\b" + re.escape(lemmatize(needle)) + r"\b")
    return bool(pattern.search(lemmatize(haystack)))


def extract_nouns(text: str) -> list:
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return [lemmatize(w) for w, t in tagged if t.startswith("NN") and len(w) > 2]


def build_gt_set(item: dict) -> set:
    gt = []
    gt.extend(item.get("L1_scene", {}).get("predicted_scene", "").lower().split())
    for f in item.get("L2_fixtures", {}).get("fixtures_detected", []):
        gt.extend(f.lower().split())
    for p in item.get("L3_products", {}).get("top_products", []):
        gt.extend(p.get("product", "").lower().split())
    for ocr in item.get("L4_attributes", {}).get("ocr_text", []):
        gt.extend(ocr.get("text", "").lower().split())

    raw = set(gt)
    if "endcap" in raw: raw.update(["shelf", "display"])
    if "till"   in raw: raw.update(["checkout", "register"])

    return {lemmatize("".join(c for c in w if c.isalpha()))
            for w in raw if len("".join(c for c in w if c.isalpha())) > 2}


def chair_score(item: dict) -> tuple:
    """Returns (hallucinated_nouns, chair_i_score, total_nouns)."""
    caption = item.get("FINAL_CAPTION", "")
    gt_set  = build_gt_set(item)
    nouns   = [n for n in extract_nouns(caption) if n not in SAFE_WORDS]

    hallucinated = [
        n for n in nouns
        if not any(word_boundary_match(n, g) or word_boundary_match(g, n) for g in gt_set)
    ]
    chair_i = round(len(hallucinated) / max(len(nouns), 1), 3)
    return hallucinated, chair_i, len(nouns)


def load_llm_scores() -> dict:
    """Load per-image LLM Judge scores if available."""
    if not os.path.exists(LLM_JUDGE_PATH):
        return {}
    with open(LLM_JUDGE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {s["image_file"]: s for s in data.get("detailed_scores", [])}


def run_comparison():
    download_nltk()

    # ── Load captions ─────────────────────────────────────────────────────
    if not os.path.exists(MOP_PATH):
        raise FileNotFoundError(f"MOP captions not found: {MOP_PATH}")
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(f"Baseline captions not found: {BASELINE_PATH}")

    with open(MOP_PATH,      "r", encoding="utf-8") as f: mop_data      = json.load(f)
    with open(BASELINE_PATH, "r", encoding="utf-8") as f: baseline_data = json.load(f)

    # Build lookup by image_file
    mop_map      = {i["image_file"]: i for i in mop_data}
    baseline_map = {i["image_file"]: i for i in baseline_data}
    llm_map      = load_llm_scores()

    common_images = sorted(set(mop_map) & set(baseline_map))
    print(f"Comparing {len(common_images)} images with both MOP and Baseline captions.")

    # ── Per-image comparison ───────────────────────────────────────────────
    results = []
    total_mop_hall, total_base_hall, total_mop_n, total_base_n = 0, 0, 0, 0
    mop_hall_sentences, base_hall_sentences = 0, 0

    for img_name in common_images:
        mop_item  = mop_map[img_name]
        base_item = baseline_map[img_name]

        # Use MOP GT set for both (baseline has empty L1-L4 stubs)
        # Inject GT into baseline item for fair comparison
        base_item_with_gt = {**base_item,
                             "L1_scene":     mop_item.get("L1_scene", {}),
                             "L2_fixtures":  mop_item.get("L2_fixtures", {}),
                             "L3_products":  mop_item.get("L3_products", {}),
                             "L4_attributes":mop_item.get("L4_attributes", {})}

        mop_hall,  mop_ci,  mop_n  = chair_score(mop_item)
        base_hall, base_ci, base_n = chair_score(base_item_with_gt)

        total_mop_hall  += len(mop_hall);  total_mop_n  += mop_n
        total_base_hall += len(base_hall); total_base_n += base_n
        if mop_hall:  mop_hall_sentences  += 1
        if base_hall: base_hall_sentences += 1

        llm = llm_map.get(img_name, {})

        results.append({
            "image": img_name,
            "mop_caption":      mop_item.get("FINAL_CAPTION", ""),
            "baseline_caption": base_item.get("FINAL_CAPTION", ""),
            "mop_hallucinated_nouns":      mop_hall,
            "baseline_hallucinated_nouns": base_hall,
            "mop_chair_i":      mop_ci,
            "baseline_chair_i": base_ci,
            "chair_i_delta":    round(base_ci - mop_ci, 3),   # positive = baseline worse
            "llm_accuracy":     llm.get("accuracy_score"),
            "llm_relevance":    llm.get("relevance_score"),
            "llm_absence":      llm.get("absence_handling_score"),
            "mop_route_cluster": mop_item.get("MOP_route_cluster"),
        })

    # ── Aggregate CHAIR ────────────────────────────────────────────────────
    n = len(common_images)
    mop_chair_i  = round(total_mop_hall  / max(total_mop_n,  1) * 100, 2)
    base_chair_i = round(total_base_hall / max(total_base_n, 1) * 100, 2)
    mop_chair_s  = round(mop_hall_sentences  / max(n, 1) * 100, 2)
    base_chair_s  = round(base_hall_sentences / max(n, 1) * 100, 2)

    # LLM Judge averages (MOP only — baseline was not judged)
    judged = [r for r in results if r["llm_accuracy"] is not None]
    avg_acc = round(sum(r["llm_accuracy"] for r in judged) / max(len(judged), 1), 2)
    avg_rel = round(sum(r["llm_relevance"] for r in judged) / max(len(judged), 1), 2)
    avg_abs = round(sum(r["llm_absence"]   for r in judged) / max(len(judged), 1), 2)

    summary = {
        "n_images_compared": n,
        "CHAIR_i": {"MOP": mop_chair_i,  "Baseline": base_chair_i,
                    "improvement_%": round(base_chair_i - mop_chair_i, 2)},
        "CHAIR_s": {"MOP": mop_chair_s,  "Baseline": base_chair_s,
                    "improvement_%": round(base_chair_s - mop_chair_s, 2)},
        "LLM_Judge_MOP_only": {"accuracy": avg_acc, "relevance": avg_rel,
                               "absence_handling": avg_abs, "n_judged": len(judged)},
        "per_image": results,
    }

    os.makedirs("data/eval_results", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"\nSaved comparison JSON -> {OUT_JSON}")

    # ── Markdown report ────────────────────────────────────────────────────
    lines = [
        "# MOP vs Vanilla Baseline — Comparison Report\n",
        f"**Images compared**: {n}  \n",
        "",
        "## 📊 Aggregate Scores\n",
        "| Metric | MOP Pipeline | Vanilla Baseline | Δ Improvement |",
        "|---|---|---|---|",
        f"| CHAIR_i ↓ | **{mop_chair_i}%** | {base_chair_i}% | **+{round(base_chair_i - mop_chair_i, 2)}pp** |",
        f"| CHAIR_s ↓ | **{mop_chair_s}%** | {base_chair_s}% | **+{round(base_chair_s - mop_chair_s, 2)}pp** |",
        "",
    ]

    if judged:
        lines += [
            "## 🤖 LLM-as-a-Judge (MOP only, 10-point scale)\n",
            "| Criterion | Score |",
            "|---|---|",
            f"| Accuracy | {avg_acc} / 10 |",
            f"| Relevance | {avg_rel} / 10 |",
            f"| Absence Handling | {avg_abs} / 10 |",
            f"| Evaluated | {len(judged)} images |",
            "",
        ]

    lines += [
        "---",
        "## 🖼️ Per-Image Caption Comparison\n",
        "> Sorted by hallucination improvement (most improved first)\n",
    ]

    sorted_results = sorted(results, key=lambda r: r["chair_i_delta"], reverse=True)

    for r in sorted_results:
        mop_h  = r["mop_hallucinated_nouns"]
        base_h = r["baseline_hallucinated_nouns"]
        mop_badge  = "✅ PASS" if not mop_h  else f"❌ FAIL ({', '.join(mop_h)})"
        base_badge = "✅ PASS" if not base_h else f"❌ FAIL ({', '.join(base_h)})"

        llm_str = ""
        if r["llm_accuracy"] is not None:
            llm_str = (f"  \n  **LLM Judge**: Accuracy {r['llm_accuracy']}/10 | "
                       f"Relevance {r['llm_relevance']}/10 | "
                       f"Absence {r['llm_absence']}/10")

        lines += [
            f"### `{r['image']}` — Cluster {r['mop_route_cluster']}",
            f"| | MOP Pipeline | Vanilla Baseline |",
            f"|---|---|---|",
            f"| CHAIR_i | {r['mop_chair_i']:.3f} | {r['baseline_chair_i']:.3f} |",
            f"| Hallucinated | {mop_badge} | {base_badge} |",
            f"",
            f"**MOP Caption:**",
            f"> {r['mop_caption'].strip()}",
            f"",
            f"**Vanilla Caption:**",
            f"> {r['baseline_caption'].strip()}",
            llm_str,
            "",
            "---",
        ]

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved markdown report  -> {OUT_REPORT}")

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  MOP vs Baseline — CHAIR Comparison")
    print("=" * 50)
    print(f"  CHAIR_i : MOP {mop_chair_i}%  vs  Baseline {base_chair_i}%"
          f"  (↓ {round(base_chair_i - mop_chair_i, 2)}pp improvement)")
    print(f"  CHAIR_s : MOP {mop_chair_s}%  vs  Baseline {base_chair_s}%"
          f"  (↓ {round(base_chair_s - mop_chair_s, 2)}pp improvement)")
    if judged:
        print(f"  LLM Judge   : Acc {avg_acc} | Rel {avg_rel} | Abs {avg_abs}  (MOP only)")
    print("=" * 50)
    print(f"\nFull report: {OUT_REPORT}")


if __name__ == "__main__":
    run_comparison()

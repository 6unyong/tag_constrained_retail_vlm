"""
pipeline_5_interim_captions.py
──────────────────────────────
Temporary caption generator that works WITHOUT Ollama or Gemini API.
Uses clustered_routes.json (already computed) to build rule-based captions
that match the final_captions.json schema expected by pipeline_6 and pipeline_7.

Covers: 2,814 images already in clustered_routes.json
Output: data/cache/final_captions_interim.json

Usage:
    python src/pipeline_5_interim_captions.py
    python src/pipeline_5_interim_captions.py --out data/cache/final_captions.json  # overwrite main
"""
import os
import json
import argparse

IN_PATH         = "data/cache/clustered_routes.json"
MOP_PROMPTS     = "data/cache/mop_prompts.json"
OUT_PATH_DEFAULT = "data/cache/final_captions_interim.json"

CLUSTER_STARTERS = {
    0: "This promotional endcap display shows",
    1: "This endcap display features",
}
FALLBACK_STARTER = "This retail display contains"


# ── helpers ────────────────────────────────────────────────────────────────────

def build_l3_str(item: dict) -> str:
    """Return deduplicated Hard/Soft product names (Absence excluded)."""
    l3_source = item.get("l3_source", "gemini")
    top = item.get("L3_products", {}).get("top_products", [])
    if l3_source == "empty" or not top:
        return "no specific products visible"
    seen, names = set(), []
    for p in top:
        if p.get("tag_type") == "Absence":
            continue
        name = p.get("product", "")
        core = " ".join(name.split()[:3]).lower()
        if core not in seen:
            names.append(name)
            seen.add(core)
    return ", ".join(names) if names else "no specific products visible"


def build_caption(item: dict) -> str:
    """
    Assemble a 2-sentence rule-based caption that mirrors what a VLM would
    produce using the MOP prompt. Follows the cluster starter convention so
    downstream Retail-CHAIR / LLM-Judge comparisons remain valid.
    """
    cluster = item.get("MOP_route_cluster", 0)
    starter = CLUSTER_STARTERS.get(cluster, FALLBACK_STARTER)

    l1   = item["L1_scene"]["predicted_scene"]
    l2   = list(item["L2_fixtures"].get("fixtures_detected", []))
    l2_u = list(dict.fromkeys(l2))            # deduplicate, preserve order
    l2_str = ", ".join(l2_u) if l2_u else "general retail fixtures"

    l3_str = build_l3_str(item)

    ctx  = item.get("global_context", [])
    ctx_str = ctx[0] if ctx else "a retail environment"

    ops   = item["L4_attributes"]["operational_state"]
    stock = ops.get("stock_level", {}).get("label", "unknown stock level")
    tidy  = ops.get("tidiness",    {}).get("label", "unknown organisation")
    promo = ops.get("promotion",   {}).get("label", "unknown promotional status")

    ocr_raw = item["L4_attributes"].get("ocr_text", [])
    # Take up to 5 high-confidence OCR tokens (≥0.80)
    ocr_hq  = [t["text"] for t in ocr_raw if t.get("confidence", 0) >= 0.80][:5]
    ocr_str = ", ".join(ocr_hq) if ocr_hq else None

    # ── Sentence 1: what, where, what's on display ──────────────────────────
    s1 = f"{starter} a {l1} fitted with {l2_str}, displaying {l3_str}."

    # ── Sentence 2: operational context + OCR ───────────────────────────────
    parts = []
    # Normalise verbose tidiness/stock labels to short tokens
    stock_short = "well-stocked"   if "well" in stock.lower()   else \
                  "low stock"      if "low"  in stock.lower()   else \
                  "out of stock"   if "out"  in stock.lower()   else \
                  "unknown stock level"
    tidy_short  = "tidy"           if "tidy" in tidy.lower() and "untidy" not in tidy.lower() else \
                  "untidy"         if "untidy" in tidy.lower() or "disorganised" in tidy.lower() or "messy" in tidy.lower() else \
                  "neatly arranged"
    promo_short = "with active promotions" if "promo" in promo.lower() and "no promo" not in promo.lower() and "ambig" not in promo.lower() else ""

    parts.append(f"The display appears {stock_short} and {tidy_short}")
    if promo_short:
        parts.append(promo_short)
    if ocr_str:
        parts.append(f"with visible text including {ocr_str}")
    parts.append(f"set within {ctx_str}.")

    s2 = ", ".join(parts[:2])
    if len(parts) > 2:
        s2 += ", " + ", ".join(parts[2:])

    return f"{s1} {s2}"


# ── main ───────────────────────────────────────────────────────────────────────

def run(out_path: str):
    print(f"[INTERIM] Loading {IN_PATH} …")
    with open(IN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INTERIM] {len(data)} images found in clustered_routes.json")

    # Checkpoint: skip already-generated entries
    existing = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                existing[entry["image_path"]] = entry
        print(f"[CHECKPOINT] {len(existing)} entries already in output — skipping.")

    results = dict(existing)
    new_count = 0

    for item in data:
        img_path = item["image_path"]
        if img_path in results:
            continue

        caption = build_caption(item)

        out_item = dict(item)               # preserve all upstream fields
        out_item["FINAL_CAPTION"] = caption
        out_item["caption_source"] = "rule_based_interim"   # flag for downstream
        results[img_path] = out_item
        new_count += 1

        # Progress feedback every 500
        if new_count % 500 == 0:
            print(f"  … {new_count} captions generated so far")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(list(results.values()), f, indent=4, ensure_ascii=False)

    print(f"\n[DONE] {new_count} new captions generated.")
    print(f"[DONE] Total in file: {len(results)}")
    print(f"[DONE] Saved → {out_path}")

    # Quick sanity check
    print("\n── Sample caption ──")
    sample = list(results.values())[0]
    print(f"  Image : {sample['image_path']}")
    print(f"  Cluster: {sample.get('MOP_route_cluster')}")
    print(f"  Caption: {sample['FINAL_CAPTION']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule-based interim caption generator")
    parser.add_argument(
        "--out", default=OUT_PATH_DEFAULT,
        help=f"Output JSON path (default: {OUT_PATH_DEFAULT})"
    )
    args = parser.parse_args()
    run(args.out)

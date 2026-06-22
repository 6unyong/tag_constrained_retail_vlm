"""
Task 17: L-CHAIR Hallucination Metric Evaluator
Extracts nouns from the generated VLM caption and compares them
against the Ground Truth Object List (L1 + L2 + L3 + OCR) to calculate
CHAIR_i (instance-level) and CHAIR_s (sentence-level) hallucination rates.

Matching Strategy (academic rigor):
- Word-boundary regex matching prevents substring false-positives
  (e.g., "cola" would incorrectly match inside "chocolate" with naive `in` check).
- NLTK WordNetLemmatizer normalises plural/inflected forms before comparison
  (e.g., "bottles" -> "bottle", "sandwiches" -> "sandwich").
"""
import os
import sys
import json
import re
import nltk

# Fix Windows console encoding (cp949 can't handle accented chars in GT words)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer


def download_nltk_data():
    for resource in [
        "punkt", "punkt_tab",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
        "wordnet", "omw-1.4",
    ]:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: could not download NLTK resource '{resource}': {e}")


lemmatizer = WordNetLemmatizer()

# Words that VLMs legitimately use to describe retail scenes without hallucinating
SAFE_WORDS = {
    "image", "photo", "variety", "types", "status", "condition", "store", "area",
    "item", "brand", "text", "visible", "product", "place", "bottle", "information",
    "retailer", "state", "sign", "shelf", "display", "section", "aisle", "report",
    "inspection", "grocery", "retail", "promotional", "signage", "region",
}


def lemmatize(word: str) -> str:
    """Return the lemmatized (base) form of a word (noun mode)."""
    return lemmatizer.lemmatize(word.lower(), pos="n")


def word_boundary_match(needle: str, haystack: str) -> bool:
    """
    True if `needle` appears as a complete word inside `haystack`.
    Uses \\b word-boundary regex to prevent false substring matches
    (e.g., 'cola' should NOT match inside 'chocolate').
    Both are lemmatized before comparison.
    """
    needle_lem = lemmatize(needle)
    haystack_lem = lemmatize(haystack)
    pattern = re.compile(r"\b" + re.escape(needle_lem) + r"\b")
    return bool(pattern.search(haystack_lem))


def extract_nouns(text: str) -> list:
    """Extracts lemmatized nouns (NN, NNS, NNP, NNPS) from a text string."""
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    nouns = [lemmatize(word) for word, tag in tagged if tag.startswith("NN")]
    return nouns


def build_ground_truth_set(item: dict) -> set:
    """Combines L1, L2, L3 and OCR into a unified set of lemmatized GT words."""
    gt_words = []

    # Scene
    scene = item.get("L1_scene", {}).get("predicted_scene", "")
    gt_words.extend(scene.lower().split())

    # Fixtures
    for fixture in item.get("L2_fixtures", {}).get("fixtures_detected", []):
        gt_words.extend(fixture.lower().split())

    # Products (all, no cap)
    for p in item.get("L3_products", {}).get("top_products", []):
        gt_words.extend(p.get("product", "").lower().split())

    # OCR strings (all, no artificial cap)
    for ocr in item.get("L4_attributes", {}).get("ocr_text", []):
        gt_words.extend(ocr.get("text", "").lower().split())

    # Synonym mappings for common retail vocabulary
    raw_set = set(gt_words)
    if "endcap" in raw_set:
        raw_set.update(["shelf", "display"])
    if "till" in raw_set:
        raw_set.update(["checkout", "register"])

    # Clean, lemmatize, and filter short/punctuation-only tokens
    valid_gt = set()
    for w in raw_set:
        w_clean = "".join(c for c in w if c.isalpha())
        if len(w_clean) > 2:
            valid_gt.add(lemmatize(w_clean))

    return valid_gt


def run_chair_evaluation():
    download_nltk_data()

    input_path = "data/cache/final_captions.json"
    if not os.path.exists(input_path):
        raise FileNotFoundError("final_captions.json not found. Run pipeline_5 first.")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_hallucinated_instances = 0
    total_caption_nouns = 0
    hallucinated_sentences = 0
    total_sentences = len(data)

    print(f"Running L-CHAIR Evaluation on {len(data)} captions...")
    eval_results = []

    for item in data:
        caption = item.get("FINAL_CAPTION", "")
        img_name = item.get("image_file", "unknown")

        gt_set = build_ground_truth_set(item)
        nouns_in_caption = extract_nouns(caption)

        # Filter trivially short or punctuation noise
        caption_nouns_clean = [n for n in nouns_in_caption if len(n) > 2]

        # Hallucination detection using word-boundary matching + lemmatization
        hallucinated_nouns = []
        for noun in caption_nouns_clean:
            if noun in SAFE_WORDS:
                continue

            # A noun is hallucinated if it does NOT match any GT word at the boundary level
            matched = any(
                word_boundary_match(noun, gt_word) or word_boundary_match(gt_word, noun)
                for gt_word in gt_set
            )
            if not matched:
                hallucinated_nouns.append(noun)

        inst_total = len(caption_nouns_clean)
        inst_hallucinated = len(hallucinated_nouns)

        total_caption_nouns += inst_total
        total_hallucinated_instances += inst_hallucinated

        if inst_hallucinated > 0:
            hallucinated_sentences += 1

        print(f"\n--- {img_name} ---")
        print(f"Caption: {caption.strip()}")
        print(f"GT Words (sample): {list(gt_set)[:8]}...")
        if hallucinated_nouns:
            print(f"[FAIL] Hallucination Detected: {hallucinated_nouns}")
        else:
            print("[PASS] No Hallucination Detected!")

        eval_results.append({
            "image": img_name,
            "hallucinated_nouns": hallucinated_nouns,
            "chair_i_score": round(inst_hallucinated / max(inst_total, 1), 3),
        })

    chair_i = round(total_hallucinated_instances / max(total_caption_nouns, 1) * 100, 2)
    chair_s = round(hallucinated_sentences / max(total_sentences, 1) * 100, 2)

    print("\n==================================")
    print(f"L-CHAIR_i (Instance Object Hallucination): {chair_i}%")
    print(f"L-CHAIR_s (Sentence Hallucination):        {chair_s}%")
    print("==================================")

    os.makedirs("data/eval_results", exist_ok=True)
    with open("data/eval_results/chair_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "CHAIR_i": chair_i,
            "CHAIR_s": chair_s,
            "matching_strategy": (
                "Word-boundary regex + WordNet lemmatization. "
                "Prevents false positives from substring matches (e.g., 'cola' vs 'chocolate'). "
                "Normalises plurals and inflected forms before comparison."
            ),
            "details": eval_results,
        }, f, indent=4)

    print("Saved results to data/eval_results/chair_metrics.json")


if __name__ == "__main__":
    run_chair_evaluation()

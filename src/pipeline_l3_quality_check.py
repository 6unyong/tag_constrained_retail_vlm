"""
L3 Tag Quality Check
====================
Randomly samples N images and renders a self-contained HTML page showing
each image alongside its L3 product tags, so the researcher can manually
verify whether the tags actually match the visual content.

Outputs: data/eval_results/l3_quality_check.html

Usage:
    python src/pipeline_l3_quality_check.py            # default 20 images
    python src/pipeline_l3_quality_check.py --n 30
    python src/pipeline_l3_quality_check.py --seed 99
"""
import os
import json
import random
import argparse
from datetime import datetime

TAGS_PATH = "data/cache/hierarchical_tags_final.json"
CAPS_PATH = "data/cache/final_captions.json"
IMG_DIR   = "data/processed"
OUT_HTML  = "l3_quality_check.html"  # output at project root so relative paths work


# ── helpers ──────────────────────────────────────────────────────────────────

# Image tag using relative path (keeps file small; HTML must be opened from project root)
def img_tag_html(img_name: str) -> str:
    rel_path = f"{IMG_DIR}/{img_name}"
    if os.path.exists(rel_path):
        return (f'<img src="{rel_path}" '
                f'style="max-width:200px;max-height:160px;object-fit:cover;border-radius:6px;" '
                f'onerror="this.style.display=\'none\'">')
    return ('<div style="width:200px;height:160px;background:#eee;'
            'display:flex;align-items:center;justify-content:center;'
            'border-radius:6px;color:#aaa">No image</div>')


def fmt_l3_tags(l3: dict) -> str:
    products = l3.get("top_products", [])
    if not products:
        return "<em style='color:#999'>No products identified</em>"
    rows = []
    for p in products[:10]:
        tt   = p.get("tag_type", "Soft")
        name = p.get("product", "")
        conf = p.get("confidence", 0)
        color = {"Hard": "#1e8449", "Soft": "#9a7d0a", "Absence": "#922b21"}.get(tt, "#555")
        badge_bg = {"Hard": "#d5f5e3", "Soft": "#fef9e7", "Absence": "#fde8e8"}.get(tt, "#eee")
        rows.append(
            f'<div style="margin:.25rem 0;font-size:.83rem;">'
            f'<span style="background:{badge_bg};color:{color};padding:.1rem .4rem;'
            f'border-radius:10px;font-size:.72rem;font-weight:700;">{tt}</span> '
            f'<b>{name}</b> <span style="color:#aaa">({conf:.2f})</span></div>'
        )
    return "".join(rows)


def fmt_prompt_info(item: dict) -> str:
    """Show what would be injected into the MOP prompt as anchors."""
    l3 = item.get("L3_products", {})
    products = l3.get("top_products", [])
    hard = [p["product"] for p in products if p.get("tag_type") == "Hard"
            and p.get("confidence", 0) >= 0.55]
    soft = [p["product"] for p in products if p.get("tag_type") == "Soft"
            and p.get("confidence", 0) >= 0.30]
    absence = [p["product"] for p in products if p.get("tag_type") == "Absence"]

    parts = []
    if hard:
        parts.append(f'<b style="color:#1e8449">Anchors (Hard)</b>: {", ".join(hard)}')
    if soft:
        parts.append(f'<b style="color:#9a7d0a">Anchors (Soft)</b>: {", ".join(soft)}')
    if absence:
        parts.append(f'<b style="color:#922b21">Withheld (Absence)</b>: {", ".join(absence)}')
    return "<br>".join(parts) if parts else "<em>No anchors extracted</em>"


def make_card(item: dict, mop_cap: str, idx: int) -> str:
    img_name = item.get("image_file", "")
    img_html = img_tag_html(img_name)

    l3_html      = fmt_l3_tags(item.get("L3_products", {}))
    l4           = item.get("L4_attributes", {})
    ops          = l4.get("operational_state", {})
    ocr_texts    = [t.get("text", "") for t in l4.get("ocr_text", [])[:6]]
    ocr_str      = ", ".join(ocr_texts) if ocr_texts else "—"
    stock        = ops.get("stock_level", {}).get("label", "Unknown")
    tidy         = ops.get("tidiness", {}).get("label", "Unknown")
    promo        = ops.get("promotion", {}).get("label", "Unknown")
    l1_scene     = item.get("L1_scene", {}).get("predicted_scene", "—")
    ctx          = ", ".join(item.get("global_context", [])) or "—"
    anchor_html  = fmt_prompt_info(item)

    mop_preview  = (mop_cap[:200] + "…") if len(mop_cap) > 200 else mop_cap

    wrong_badge  = ('<span id="wrong-{idx}" style="display:none;background:#fde8e8;color:#c0392b;'
                    'padding:.2rem .7rem;border-radius:12px;font-size:.75rem;font-weight:700;">'
                    '❌ L3 Tags Seem Wrong</span>')
    ok_badge     = ('<span id="ok-{idx}" style="display:none;background:#d5f5e3;color:#1e8449;'
                    'padding:.2rem .7rem;border-radius:12px;font-size:.75rem;font-weight:700;">'
                    '✅ L3 Tags OK</span>')

    return f"""
<div class="card" id="card-{idx}">
  <div class="card-header">
    <span class="img-name">#{idx+1} &nbsp; {img_name}</span>
    <div style="display:flex;gap:.5rem;align-items:center;">
      <button onclick="markOk({idx})"
        style="background:#d5f5e3;color:#1e8449;border:none;border-radius:8px;
               padding:.3rem .8rem;cursor:pointer;font-size:.8rem;font-weight:600;">
        ✅ Tags OK
      </button>
      <button onclick="markWrong({idx})"
        style="background:#fde8e8;color:#c0392b;border:none;border-radius:8px;
               padding:.3rem .8rem;cursor:pointer;font-size:.8rem;font-weight:600;">
        ❌ Tags Wrong
      </button>
      <span id="badge-{idx}" style="font-size:.75rem;color:#999">Not reviewed</span>
    </div>
  </div>
    <div class="card-body">
    <div class="col-img">{img_html}</div>
    <div class="col-tags">
      <div class="section-label">L0 Context</div>
      <div style="font-size:.82rem;color:#555;margin-bottom:.5rem">{ctx}</div>
      <div class="section-label">L1 Scene</div>
      <div style="font-size:.82rem;color:#555;margin-bottom:.5rem">{l1_scene}</div>
      <div class="section-label">L3 Product Tags</div>
      {l3_html}
      <div class="section-label" style="margin-top:.5rem">L4 Attributes</div>
      <div style="font-size:.8rem;color:#555">
        Stock: <b>{stock}</b> &nbsp;|&nbsp; Tidy: <b>{tidy}</b> &nbsp;|&nbsp; Promo: <b>{promo}</b><br>
        OCR: <span style="color:#888">{ocr_str}</span>
      </div>
    </div>
    <div class="col-prompt">
      <div class="section-label">MOP Prompt Anchors</div>
      <div style="font-size:.8rem;line-height:1.6;margin-bottom:.6rem">{anchor_html}</div>
      <div class="section-label">Generated MOP Caption</div>
      <div style="font-size:.82rem;line-height:1.6;color:#333;font-style:italic">"{mop_preview}"</div>
    </div>
  </div>
</div>"""


CSS = """
:root { --kcl:#003D79; --kcl-pale:#e8f0fb; --border:#dde3ef; }
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:'Segoe UI',system-ui,sans-serif; background:#f6f8fc; color:#1a1a2e; }
.hero { background:linear-gradient(135deg,#003D79 0%,#1a6cb5 100%);
        color:#fff; padding:2.5rem 2rem 2rem; text-align:center; }
.hero h1 { font-size:1.8rem; font-weight:700; }
.hero .sub { font-size:.95rem; opacity:.8; margin-top:.5rem; }
.summary-box { max-width:900px; margin:1.5rem auto; padding:1rem 1.5rem;
               background:#fff; border:1px solid var(--border); border-radius:10px;
               display:flex; gap:2rem; flex-wrap:wrap; }
.stat { text-align:center; }
.stat .val { font-size:2rem; font-weight:700; color:var(--kcl); }
.stat .lbl { font-size:.78rem; color:#777; text-transform:uppercase; }
.container { max-width:1150px; margin:0 auto; padding:1.5rem; }
.card { background:#fff; border:1px solid var(--border); border-radius:12px;
        margin-bottom:1.5rem; overflow:hidden;
        box-shadow:0 2px 10px rgba(0,0,0,.06); transition:box-shadow .2s; }
.card:hover { box-shadow:0 4px 18px rgba(0,0,0,.1); }
.card-header { background:var(--kcl-pale); padding:.6rem 1rem;
               display:flex; align-items:center; justify-content:space-between;
               border-bottom:1px solid var(--border); }
.img-name { font-weight:600; color:var(--kcl); font-size:.9rem; }
.card-body { display:grid; grid-template-columns:210px 1fr 1.2fr; gap:0; }
.col-img   { padding:.8rem; background:#fafafa; border-right:1px solid var(--border);
              display:flex; align-items:flex-start; justify-content:center; padding-top:1rem; }
.col-tags  { padding:1rem; border-right:1px solid var(--border); }
.col-prompt { padding:1rem; }
.section-label { font-size:.65rem; font-weight:700; text-transform:uppercase;
                 letter-spacing:1px; color:var(--kcl); margin-bottom:.3rem; }
footer { text-align:center; padding:2rem; color:#999; font-size:.8rem;
         border-top:1px solid var(--border); margin-top:2rem; }
@media(max-width:800px){
  .card-body { grid-template-columns:1fr; }
  .col-img { display:none; }
}
"""

JS = """
const results = {};
function markOk(idx) {
  results[idx] = 'ok';
  document.getElementById('badge-' + idx).innerHTML
    = '<span style="color:#1e8449;font-weight:700">✅ OK</span>';
  updateSummary();
}
function markWrong(idx) {
  results[idx] = 'wrong';
  document.getElementById('badge-' + idx).innerHTML
    = '<span style="color:#c0392b;font-weight:700">❌ Wrong</span>';
  updateSummary();
}
function updateSummary() {
  const vals = Object.values(results);
  const ok    = vals.filter(v => v === 'ok').length;
  const wrong = vals.filter(v => v === 'wrong').length;
  document.getElementById('stat-ok').textContent    = ok;
  document.getElementById('stat-wrong').textContent = wrong;
  document.getElementById('stat-left').textContent  =
    parseInt(document.getElementById('stat-total').textContent) - ok - wrong;
  const pct = vals.length > 0 ? Math.round(ok / vals.length * 100) : '—';
  document.getElementById('stat-pct').textContent = pct + (vals.length > 0 ? '%' : '');
}
"""


def run():
    parser = argparse.ArgumentParser(description="L3 Tag Quality Check — visual review tool")
    parser.add_argument("--n",    type=int, default=20, help="Number of images to sample (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out",  default=OUT_HTML,    help=f"Output HTML path (default: {OUT_HTML})")
    args = parser.parse_args()

    if not os.path.exists(TAGS_PATH):
        raise FileNotFoundError(f"Tag cache not found: {TAGS_PATH}")
    if not os.path.exists(CAPS_PATH):
        raise FileNotFoundError(f"MOP captions not found: {CAPS_PATH}")

    with open(TAGS_PATH, "r", encoding="utf-8") as f:
        tags_raw = json.load(f)
    with open(CAPS_PATH, "r", encoding="utf-8") as f:
        caps_raw = json.load(f)

    # Build lookup
    tags_map = {}
    if isinstance(tags_raw, list):
        tags_map = {item.get("image_file", ""): item for item in tags_raw if "image_file" in item}
    elif isinstance(tags_raw, dict):
        # keyed by path or filename
        for k, v in tags_raw.items():
            fn = os.path.basename(k)
            tags_map[fn] = v

    caps_map = {item["image_file"]: item.get("FINAL_CAPTION", "") for item in caps_raw
                if "image_file" in item}

    # Intersect: images that have both tags and MOP captions
    common = sorted(set(tags_map.keys()) & set(caps_map.keys()))
    print(f"[L3-CHECK] {len(tags_map)} tagged | {len(caps_map)} captioned | {len(common)} common")

    random.seed(args.seed)
    sample = random.sample(common, min(args.n, len(common)))
    print(f"[L3-CHECK] Sampling {len(sample)} images (seed={args.seed})")

    cards_html = ""
    for idx, img_name in enumerate(sorted(sample)):
        item    = tags_map[img_name]
        mop_cap = caps_map.get(img_name, "")
        cards_html += make_card(item, mop_cap, idx)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>L3 Tag Quality Check — KCL Dissertation</title>
  <style>{CSS}</style>
</head>
<body>
<div class="hero">
  <h1>🔍 L3 Tag Quality Check</h1>
  <div class="sub">Manual verification: Do the L3 product tags match the actual image?
    &nbsp;|&nbsp; Sample: {len(sample)} images &nbsp;|&nbsp; Generated: {now}
  </div>
</div>

<div class="summary-box">
  <div class="stat"><div class="val" id="stat-total">{len(sample)}</div><div class="lbl">Total</div></div>
  <div class="stat"><div class="val" id="stat-ok" style="color:#1e8449">0</div><div class="lbl">Tags OK</div></div>
  <div class="stat"><div class="val" id="stat-wrong" style="color:#c0392b">0</div><div class="lbl">Tags Wrong</div></div>
  <div class="stat"><div class="val" id="stat-left">{len(sample)}</div><div class="lbl">Not Reviewed</div></div>
  <div class="stat"><div class="val" id="stat-pct">—</div><div class="lbl">Tag Accuracy</div></div>
  <div style="font-size:.83rem;color:#555;align-self:center;max-width:350px;">
    Click <b>✅ Tags OK</b> or <b>❌ Tags Wrong</b> for each image.
    Tag accuracy = proportion of images where L3 tags are visually justified.
    Record the final % in dissertation Limitations section.
  </div>
</div>

<div class="container">
  {cards_html}
</div>

<footer>
  KCL MSc Dissertation &nbsp;|&nbsp; L3 Tag Quality Verification Tool &nbsp;|&nbsp;
  Tags sourced from pipeline_3b_l3_product_tagging.py (Gemini + CLIP)
</footer>

<script>{JS}</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    abs_path = os.path.abspath(args.out)
    print(f"[DONE] L3 quality check report -> {abs_path}")
    print(f"       Open this file in your browser (must open from project root for images to load):")
    print(f"       file:///{abs_path.replace(chr(92), '/')}") 
    print(f"       Then click Tags OK / Tags Wrong for each image.")


if __name__ == "__main__":
    run()

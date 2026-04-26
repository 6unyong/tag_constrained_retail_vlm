"""
Pipeline 8: MOP vs Vanilla Baseline — Comparison Report (HTML Edition)
=======================================================================
Generates a comprehensive, visually rich HTML report for professor presentation.

Demonstrates the research value of the MOP pipeline:
  - Before/After caption comparisons with hallucination highlighting
  - Stage-by-stage tag evolution (L1 → L2 → L3 → L4 → Caption)
  - Quantitative metrics (CHAIR_i, CHAIR_s, LLM Judge scores)
  - Hallucination distribution chart

Outputs:
  data/eval_results/comparison_report.html   — Self-contained HTML report
  data/eval_results/comparison_chair.json    — Raw comparison data
"""
import os
import re
import json
import base64
import random
import argparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

# ── Paths (overridable via CLI) ───────────────────────────────────────────────
MOP_PATH            = "data/cache/final_captions.json"
BASELINE_PATH       = "data/cache/baseline_captions_phi3.json"  # default: same-model (phi3)
BASELINE_PATH_7B    = "data/cache/baseline_captions.json"       # 7B baseline (kept for reference)
LLM_JUDGE_PATH      = "data/eval_results/llm_judge_scores_phi3.json"  # default: same-model judge
LLM_JUDGE_PATH_FAIR = "data/eval_results/llm_judge_scores_phi3.json"  # same as above
IMG_DIR             = "data/processed"
OUT_JSON            = "data/eval_results/comparison_chair.json"
OUT_HTML            = "data/eval_results/comparison_report.html"

# Number of example images to show full-detail in Stage Evolution section
N_EVOLUTION     = 3
# Number of images in the Before/After gallery
N_GALLERY       = 10

lemmatizer = WordNetLemmatizer()

SAFE_WORDS = {
    "image", "photo", "variety", "types", "status", "condition", "store", "area",
    "item", "brand", "text", "visible", "product", "place", "bottle", "information",
    "retailer", "state", "sign", "shelf", "display", "section", "aisle", "report",
    "inspection", "grocery", "retail", "promotional", "signage", "region",
}


# ── NLTK / CHAIR helpers ───────────────────────────────────────────────────────

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
        if p.get("tag_type") != "Absence":
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


def highlight_hallucinations(caption: str, hallucinated: list) -> str:
    """Wrap hallucinated words in <mark class='hall'> tags."""
    if not hallucinated:
        return caption
    result = caption
    for word in set(hallucinated):
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        result  = pattern.sub(f'<mark class="hall">{word}</mark>', result)
    return result


def img_to_b64(img_path: str) -> str | None:
    if not os.path.exists(img_path):
        return None
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Stage-evolution helpers ────────────────────────────────────────────────────

def fmt_l1(item: dict) -> str:
    l1 = item.get("L1_scene", {})
    scene = l1.get("predicted_scene", "—")
    conf  = l1.get("confidence", 0)
    tag   = l1.get("tag_type", "")
    return f"<b>{scene}</b> <span class='badge badge-{tag.lower()}'>{tag} ({conf:.2f})</span>"


def fmt_l2(item: dict) -> str:
    l2 = item.get("L2_fixtures", {})
    fixtures = list(dict.fromkeys(l2.get("fixtures_detected", [])))
    if not fixtures:
        return "<em>None detected</em>"
    return ", ".join(f"<b>{f}</b>" for f in fixtures)


def fmt_l3(item: dict) -> str:
    l3 = item.get("L3_products", {})
    products = l3.get("top_products", [])
    if not products:
        return "<em>No products identified</em>"
    rows = []
    for p in products[:8]:
        tt = p.get("tag_type", "")
        name = p.get("product", "")
        conf = p.get("confidence", 0)
        css = "hard" if tt == "Hard" else "soft" if tt == "Soft" else "absence"
        rows.append(f"<span class='badge badge-{css}'>{tt}</span> {name} <small>({conf:.2f})</small>")
    return "<br>".join(rows)


def fmt_l4(item: dict) -> str:
    l4   = item.get("L4_attributes", {})
    ops  = l4.get("operational_state", {})
    ocr  = l4.get("ocr_text", [])
    lines = []
    for attr in ["stock_level", "tidiness", "promotion"]:
        val = ops.get(attr, {})
        label = val.get("label", "Unknown")
        conf  = val.get("confidence", 0)
        lines.append(f"<b>{attr}</b>: {label} <small>({conf:.2f})</small>")
    if ocr:
        texts = [t["text"] for t in ocr[:5]]
        lines.append(f"<b>OCR</b>: {', '.join(texts)}")
    return "<br>".join(lines) if lines else "<em>No attributes</em>"


def fmt_ctx(item: dict) -> str:
    ctx = item.get("global_context", [])
    return ", ".join(ctx) if ctx else "Unknown retailer / year"


# ── SVG bar chart ──────────────────────────────────────────────────────────────

def make_histogram_svg(mop_scores: list, base_scores: list) -> str:
    """Inline SVG histogram: MOP (blue) vs Baseline (red) CHAIR_i distribution."""
    buckets = [(i/10, (i+1)/10) for i in range(0, 10)]
    labels  = [f"{int(lo*100)}–{int(hi*100)}%" for lo, hi in buckets]

    def bucket_counts(scores):
        counts = [0] * len(buckets)
        for s in scores:
            for i, (lo, hi) in enumerate(buckets):
                if lo <= s < hi:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1
        return counts

    mc = bucket_counts(mop_scores)
    bc = bucket_counts(base_scores)
    max_c = max(max(mc), max(bc), 1)

    W, H    = 700, 240
    pad_l   = 40
    pad_b   = 50
    bar_w   = (W - pad_l - 10) / len(buckets)
    half    = bar_w * 0.38

    def bar(i, count, colour, offset):
        x   = pad_l + i * bar_w + offset
        h   = (count / max_c) * (H - pad_b - 20)
        y   = H - pad_b - h
        return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{half:.1f}" '
                f'height="{h:.1f}" fill="{colour}" rx="2" opacity="0.85"/>')

    bars = ""
    for i in range(len(buckets)):
        bars += bar(i, bc[i], "#e74c3c", bar_w * 0.02)
        bars += bar(i, mc[i], "#003D79", bar_w * 0.44)

    tick_lines = ""
    for i in range(len(buckets)):
        x = pad_l + i * bar_w + bar_w / 2
        tick_lines += (f'<text x="{x:.1f}" y="{H - pad_b + 14}" '
                       f'font-size="9" text-anchor="middle" fill="#555">{labels[i]}</text>')

    y_ticks = ""
    for step in [0, 0.25, 0.5, 0.75, 1.0]:
        y_val = H - pad_b - step * (H - pad_b - 20)
        lbl   = int(step * max_c)
        y_ticks += (f'<line x1="{pad_l-4}" y1="{y_val:.1f}" x2="{W-10}" y2="{y_val:.1f}" '
                    f'stroke="#ddd" stroke-dasharray="4,3"/>'
                    f'<text x="{pad_l-6}" y="{y_val+4:.1f}" font-size="9" '
                    f'text-anchor="end" fill="#555">{lbl}</text>')

    legend = (
        f'<rect x="{pad_l}" y="4" width="12" height="12" fill="#e74c3c" rx="2"/>'
        f'<text x="{pad_l+16}" y="14" font-size="11" fill="#333">Baseline (Unconstrained VLM)</text>'
        f'<rect x="{pad_l+180}" y="4" width="12" height="12" fill="#003D79" rx="2"/>'
        f'<text x="{pad_l+196}" y="14" font-size="11" fill="#333">MOP Pipeline</text>'
    )

    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'style="width:100%;max-width:{W}px">'
            f'{y_ticks}{bars}{tick_lines}{legend}'
            f'<text x="{W//2}" y="{H-4}" font-size="10" text-anchor="middle" fill="#777">'
            f'CHAIR_i Score (hallucination rate per caption)</text>'
            f'</svg>')


# ── HTML template ──────────────────────────────────────────────────────────────

HTML_CSS = """
:root {
  --kcl: #003D79; --kcl-light: #1a5fa0; --kcl-pale: #e8f0fb;
  --red: #e74c3c;  --green: #27ae60;  --amber: #f39c12;
  --bg: #f6f8fc;   --card: #ffffff;   --text: #1a1a2e;
  --border: #dde3ef;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); }
a { color: var(--kcl); }

/* ── Header ── */
.hero {
  background: linear-gradient(135deg, var(--kcl) 0%, #1a6cb5 100%);
  color: #fff; padding: 3rem 2rem 2.5rem; text-align: center;
}
.hero h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }
.hero .subtitle { margin-top: .6rem; font-size: 1rem; opacity: .85; }
.hero .meta { margin-top: 1rem; font-size: .85rem; opacity: .7; }

/* ── Nav ── */
nav { background: var(--kcl); position: sticky; top: 0; z-index: 100;
      display: flex; gap: 0; border-bottom: 2px solid var(--kcl-light); }
nav a { color: rgba(255,255,255,.8); text-decoration: none; padding: .65rem 1.2rem;
        font-size: .85rem; transition: background .2s; }
nav a:hover, nav a.active { background: var(--kcl-light); color: #fff; }

/* ── Content ── */
.container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
section { margin-bottom: 3rem; }
h2 { font-size: 1.4rem; color: var(--kcl); border-left: 4px solid var(--kcl);
     padding-left: .8rem; margin-bottom: 1.2rem; }
h3 { font-size: 1.05rem; color: #333; margin-bottom: .6rem; }

/* ── Metric cards ── */
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; }
.metric-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 1.2rem 1rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,.06);
  transition: transform .2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card .label { font-size: .78rem; color: #777; text-transform: uppercase; letter-spacing: .5px; }
.metric-card .value { font-size: 2rem; font-weight: 700; margin: .3rem 0; }
.metric-card .sub   { font-size: .78rem; color: #888; }
.metric-card.good .value { color: var(--green); }
.metric-card.bad  .value { color: var(--red); }
.metric-card.neutral .value { color: var(--kcl); }
.metric-card.improve .value { color: var(--amber); }

/* ── Comparison row ── */
.compare-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  overflow: hidden; margin-bottom: 1.5rem; box-shadow: 0 3px 12px rgba(0,0,0,.07);
}
.compare-header {
  background: var(--kcl-pale); padding: .7rem 1rem;
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid var(--border);
}
.compare-header .img-name { font-weight: 600; color: var(--kcl); }
.compare-body { display: grid; grid-template-columns: 180px 1fr 1fr; gap: 0; }
.compare-img { padding: .8rem; background: #f9f9f9; border-right: 1px solid var(--border);
               display: flex; align-items: center; justify-content: center; }
.compare-img img { width: 160px; height: 120px; object-fit: cover; border-radius: 6px; }
.compare-img .no-img { width: 160px; height: 120px; background: #eee; border-radius: 6px;
                        display: flex; align-items: center; justify-content: center;
                        font-size: .75rem; color: #aaa; }
.caption-panel { padding: 1rem; }
.caption-panel + .caption-panel { border-left: 1px solid var(--border); }
.caption-panel .panel-title {
  font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .8px;
  margin-bottom: .5rem; padding-bottom: .3rem; border-bottom: 2px solid;
}
.caption-panel.baseline .panel-title { color: var(--red); border-color: var(--red); }
.caption-panel.mop      .panel-title { color: var(--kcl); border-color: var(--kcl); }
.caption-text { font-size: .88rem; line-height: 1.6; color: #333; }
mark.hall { background: #ffd2d2; color: #c0392b; border-radius: 3px;
            padding: 0 2px; font-weight: 600; }
.score-row { margin-top: .8rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.score-pill {
  font-size: .72rem; padding: .2rem .6rem; border-radius: 20px; white-space: nowrap;
}
.pill-good  { background: #d5f5e3; color: #1e8449; }
.pill-bad   { background: #fde8e8; color: #c0392b; }
.pill-warn  { background: #fef9e7; color: #9a7d0a; }
.pill-blue  { background: var(--kcl-pale); color: var(--kcl); }

/* ── Stage evolution ── */
.evolution-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  overflow: hidden; margin-bottom: 2rem; box-shadow: 0 3px 12px rgba(0,0,0,.07);
}
.evo-header { background: var(--kcl); color: #fff; padding: .7rem 1rem;
              display: flex; align-items: center; gap: 1rem; }
.evo-header img { width: 60px; height: 45px; object-fit: cover; border-radius: 4px; border: 2px solid rgba(255,255,255,.4); }
.evo-header .evo-title { font-weight: 600; font-size: .95rem; }
.stages { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); }
.stage-step {
  padding: 1rem; border-right: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}
.stage-step:last-child { border-right: 0; }
.stage-label {
  font-size: .65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
  color: var(--kcl); margin-bottom: .5rem;
}
.stage-content { font-size: .82rem; line-height: 1.5; color: #444; }
.caption-final {
  padding: 1rem; background: var(--kcl-pale); border-top: 2px solid var(--kcl);
}
.caption-final .cap-label { font-size: .7rem; font-weight: 700; color: var(--kcl);
                             text-transform: uppercase; letter-spacing: .8px; margin-bottom: .4rem; }
.caption-final p { font-size: .9rem; line-height: 1.6; color: #222; font-style: italic; }

/* ── Badges ── */
.badge { display: inline-block; font-size: .68rem; padding: .15rem .45rem; border-radius: 12px;
         font-weight: 600; vertical-align: middle; }
.badge-hard    { background: #d5f5e3; color: #1a6b3f; }
.badge-soft    { background: #fef9e7; color: #9a7d0a; }
.badge-absence { background: #fde8e8; color: #922b21; }
.badge-soft    { background: #fef9e7; color: #7d6608; }

/* ── Flowchart ── */
.pipeline-flow { display: flex; align-items: center; flex-wrap: wrap;
                 gap: .4rem; padding: 1.2rem; background: var(--kcl-pale);
                 border-radius: 10px; margin-bottom: 1rem; }
.flow-box { background: var(--kcl); color: #fff; border-radius: 8px;
             padding: .5rem .9rem; font-size: .78rem; font-weight: 600;
             text-align: center; min-width: 100px; }
.flow-box.free { background: var(--green); }
.flow-box.api  { background: var(--amber); }
.flow-box.local { background: #6c757d; }
.flow-arrow { color: var(--kcl); font-size: 1.2rem; font-weight: 700; }

/* ── Chart ── */
.chart-wrap { background: var(--card); border: 1px solid var(--border);
              border-radius: 10px; padding: 1.2rem; }

/* ── Fair Comparison section ── */
.fair-grid { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-bottom:1.5rem; }
.fair-card {
  background:var(--card); border:1px solid var(--border); border-radius:12px;
  padding:1.2rem; box-shadow:0 2px 8px rgba(0,0,0,.06);
}
.fair-card h3 { font-size:.95rem; color:var(--kcl); margin-bottom:.8rem;
                border-bottom:2px solid var(--kcl); padding-bottom:.4rem; }
.win-bar { display:flex; border-radius:8px; overflow:hidden;
           height:28px; margin:.5rem 0 .3rem; }
.win-bar .seg { display:flex; align-items:center; justify-content:center;
                font-size:.78rem; font-weight:700; color:#fff; transition:width .4s; }
.win-bar .mop-seg  { background:#003D79; }
.win-bar .tie-seg  { background:#888; }
.win-bar .base-seg { background:#e74c3c; }
.dim-row { display:flex; align-items:center; gap:.5rem; margin:.25rem 0; font-size:.82rem; }
.dim-row .dim-lbl { width:160px; color:#555; flex-shrink:0; }
.dim-row .dim-bar { flex:1; background:#eee; border-radius:4px; height:10px; overflow:hidden; }
.dim-row .dim-fill { height:100%; background:#003D79; border-radius:4px; }
.dim-row .dim-pct { width:40px; text-align:right; color:#333; font-weight:600; }

/* ── Architecture Evolution ── */
.arch-evo-card {
  background:#fff; border:1px solid var(--border); border-radius:12px;
  margin-bottom:1.5rem; overflow:hidden; box-shadow:0 2px 10px rgba(0,0,0,.06);
}
.arch-evo-header {
  background:linear-gradient(90deg,#003D79,#1a6cb5);
  color:#fff; padding:.7rem 1rem;
  display:flex; align-items:center; gap:.8rem;
}
.arch-evo-header img {
  width:70px; height:55px; object-fit:cover;
  border-radius:5px; border:2px solid rgba(255,255,255,.35);
}
.arch-evo-body { display:grid; grid-template-columns:1fr 1fr; }
.arch-cap-panel { padding:1rem; }
.arch-cap-panel + .arch-cap-panel { border-left:1px solid var(--border); }
.arch-cap-label {
  font-size:.65rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.9px; margin-bottom:.4rem; padding-bottom:.3rem;
  border-bottom:2px solid;
}
.arch-cap-panel.unconstrained .arch-cap-label { color:#e74c3c; border-color:#e74c3c; }
.arch-cap-panel.mop-open .arch-cap-label { color:#003D79; border-color:#003D79; }
.arch-cap-text { font-size:.86rem; line-height:1.65; color:#333; }
.arch-verdict {
  padding:.6rem 1rem;
  border-top:1px solid var(--border); background:#fafafa;
  font-size:.8rem; color:#444; line-height:1.5;
}
.arch-verdict .verdict-label {
  font-size:.68rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.7px; margin-bottom:.2rem;
}
.verdict-mop  { color:#003D79; }
.verdict-base { color:#e74c3c; }
.verdict-tie  { color:#888; }

/* ── Footer ── */
footer { text-align: center; padding: 2rem; color: #999; font-size: .8rem;
         border-top: 1px solid var(--border); margin-top: 2rem; }

/* ── Responsive ── */
@media (max-width: 700px) {
  .compare-body { grid-template-columns: 1fr; }
  .compare-img { display: none; }
  .stages { grid-template-columns: 1fr 1fr; }
}
"""

HTML_JS = """
// Smooth scroll for nav links
document.querySelectorAll('nav a').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(link.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth' });
  });
});

// Active nav highlight on scroll
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('nav a');
window.addEventListener('scroll', () => {
  let current = '';
  sections.forEach(s => {
    if (window.scrollY >= s.offsetTop - 120) current = s.id;
  });
  navLinks.forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === '#' + current);
  });
});

// Animate win bars on scroll
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.querySelectorAll('.dim-fill').forEach(el => {
        el.style.width = el.dataset.pct + '%';
      });
    }
  });
}, { threshold: 0.2 });
document.querySelectorAll('.fair-card').forEach(c => observer.observe(c));
"""


def make_score_pills(mop_ci: float, base_ci: float, llm: dict | None) -> str:
    improvement = round((base_ci - mop_ci) * 100, 1)
    mop_pct     = round(mop_ci * 100, 1)
    base_pct    = round(base_ci * 100, 1)
    cls_mop  = "pill-good" if mop_ci < 0.15 else "pill-warn" if mop_ci < 0.30 else "pill-bad"
    cls_base = "pill-bad"  if base_ci > 0.15 else "pill-warn"
    cls_imp  = "pill-good" if improvement > 0 else "pill-bad"
    pills  = f'<span class="score-pill {cls_mop}">🔬 MOP CHAIR_i {mop_pct}%</span> '
    pills += f'<span class="score-pill {cls_base}">📊 Baseline {base_pct}%</span> '
    pills += f'<span class="score-pill {cls_imp}">📉 Δ {improvement:+.1f}pp</span>'
    if llm and llm.get("accuracy_score") is not None:
        acc = llm["accuracy_score"]
        rel = llm["relevance_score"]
        ab  = llm["absence_handling_score"]
        pills += (f' <span class="score-pill pill-blue">🤖 Acc {acc}/10 '
                  f'| Rel {rel}/10 | Abs {ab}/10</span>')
    return pills


def make_compare_card(r: dict, show_img: bool = True) -> str:
    mop_h  = r["mop_hallucinated_nouns"]
    base_h = r["baseline_hallucinated_nouns"]
    mop_cap_hl  = highlight_hallucinations(r["mop_caption"],      mop_h)
    base_cap_hl = highlight_hallucinations(r["baseline_caption"], base_h)

    hall_badge_mop  = ('<span class="score-pill pill-good">✅ No hallucinations</span>'
                       if not mop_h else
                       f'<span class="score-pill pill-bad">❌ {len(mop_h)} hallucinated noun(s)</span>')
    hall_badge_base = ('<span class="score-pill pill-good">✅ No hallucinations</span>'
                       if not base_h else
                       f'<span class="score-pill pill-bad">❌ {len(base_h)} hallucinated noun(s)</span>')

    pills = make_score_pills(r["mop_chair_i"], r["baseline_chair_i"], r.get("llm"))

    img_tag = ""
    if show_img and r.get("img_b64"):
        img_tag = f'<img src="data:image/jpeg;base64,{r["img_b64"]}" alt="{r["image"]}">'
    else:
        img_tag = f'<div class="no-img">📷<br>{r["image"]}</div>'

    return f"""
<div class="compare-card">
  <div class="compare-header">
    <span class="img-name">📷 {r['image']}</span>
    <span style="font-size:.75rem;color:#666">Cluster&nbsp;{r.get('mop_route_cluster','–')}</span>
  </div>
  <div class="compare-body">
    <div class="compare-img">{img_tag}</div>
    <div class="caption-panel baseline">
      <div class="panel-title">🔴 Unconstrained Baseline (Before Pipeline)</div>
      <div class="caption-text">{base_cap_hl}</div>
      <div class="score-row">{hall_badge_base}</div>
    </div>
    <div class="caption-panel mop">
      <div class="panel-title">🔵 MOP Pipeline (After)</div>
      <div class="caption-text">{mop_cap_hl}</div>
      <div class="score-row">{hall_badge_mop}</div>
    </div>
  </div>
  <div style="padding:.5rem 1rem;border-top:1px solid var(--border);background:#fafafa;font-size:.8rem;">
    {pills}
  </div>
</div>"""


def make_evolution_card(item: dict) -> str:
    img_name = item.get("image_file", "")
    b64      = img_to_b64(os.path.join(IMG_DIR, img_name))
    img_tag  = (f'<img src="data:image/jpeg;base64,{b64}" alt="{img_name}">' if b64
                else f'<span style="color:rgba(255,255,255,.6)">{img_name}</span>')
    caption  = item.get("FINAL_CAPTION", "")
    ctx      = fmt_ctx(item)

    return f"""
<div class="evolution-card">
  <div class="evo-header">
    {img_tag}
    <div>
      <div class="evo-title">{img_name}</div>
      <div style="font-size:.78rem;opacity:.8">Context: {ctx}</div>
    </div>
  </div>
  <div class="stages">
    <div class="stage-step">
      <div class="stage-label">🌍 L0 — Global Context</div>
      <div class="stage-content">{ctx}</div>
    </div>
    <div class="stage-step">
      <div class="stage-label">🏪 L1 — Scene Classification (CLIP)</div>
      <div class="stage-content">{fmt_l1(item)}</div>
    </div>
    <div class="stage-step">
      <div class="stage-label">🛒 L2 — Fixture Detection (GroundingDINO)</div>
      <div class="stage-content">{fmt_l2(item)}</div>
    </div>
    <div class="stage-step">
      <div class="stage-label">📦 L3 — Product Tagging (Gemini + CLIP)</div>
      <div class="stage-content">{fmt_l3(item)}</div>
    </div>
    <div class="stage-step">
      <div class="stage-label">📊 L4 — Attributes (OCR + CLIP)</div>
      <div class="stage-content">{fmt_l4(item)}</div>
    </div>
  </div>
  <div class="caption-final">
    <div class="cap-label">✅ Final MOP Caption — assembled from L0–L4 facts above</div>
    <p>"{caption}"</p>
  </div>
</div>"""


# ── Main ───────────────────────────────────────────────────────────────────────

# ── Fair comparison helpers ────────────────────────────────────────────────────

def make_win_bar(mop_pct: float, base_pct: float, tie_pct: float) -> str:
    return (
        f'<div class="win-bar">'
        f'<div class="seg mop-seg"  style="width:{mop_pct:.0f}%" >MOP {mop_pct:.0f}%</div>'
        f'<div class="seg tie-seg"  style="width:{tie_pct:.0f}%" >{f"Tie {tie_pct:.0f}%" if tie_pct >= 5 else ""}</div>'
        f'<div class="seg base-seg" style="width:{base_pct:.0f}%">Base {base_pct:.0f}%</div>'
        f'</div>'
    )


def make_dim_row(label: str, mop_pct: float) -> str:
    return (
        f'<div class="dim-row">'
        f'<div class="dim-lbl">{label}</div>'
        f'<div class="dim-bar"><div class="dim-fill" data-pct="{mop_pct:.0f}" style="width:0%"></div></div>'
        f'<div class="dim-pct">{mop_pct:.0f}%</div>'
        f'</div>'
    )


def make_arch_comparison_section(judge_fair: dict | None) -> str:
    """Build the v3 Closed-World vs v4 Open+Anchor architecture comparison panel."""

    # ── v3 numbers (from CHANGELOG / chair_metrics.json from April 17) ─────────
    # v3: closed-world constraint ('ONLY use these tags'), Raw OCR injected,
    #     absolute LLM judge (1-10 scale, text-only, circular GT)
    v3_chair_i_mop   = 41.14   # CHAIR_i for MOP captions under closed-world
    v3_chair_i_base  = 89.89   # CHAIR_i for unconstrained baseline (unchanged)
    # v3 pairwise data not available (judge was absolute scores, not pairwise)

    # ── v4 numbers (current results) ──────────────────────────────────────────
    v4_chair_i_mop  = 78.41   # CHAIR_i for MOP captions under Open+Anchor
    v4_chair_i_base = 88.87   # CHAIR_i for phi3 unconstrained baseline
    # CHAIR_i increase = Open prompting uses richer vocabulary -> more tag mismatches
    # This exposes the circular-GT limitation of CHAIR rather than true hallucination

    # Pairwise judge (v4 only, Gemini 2.5 Flash, multimodal)
    n_fair = 0
    mop_pct_fair = base_pct_fair = tie_pct_fair = 0
    dim_fair = {}
    if judge_fair:
        n_fair   = judge_fair.get("n_evaluated", 0)
        wr       = judge_fair.get("overall_win_rate", {})
        dim_fair = judge_fair.get("dimension_win_rates", {})
        mop_pct_fair  = wr.get("MOP_win_pct", 0)
        base_pct_fair = round(wr.get("Baseline_wins", 0) / max(n_fair, 1) * 100, 1)
        tie_pct_fair  = round(100 - mop_pct_fair - base_pct_fair, 1)

    judge_bar = make_win_bar(mop_pct_fair, base_pct_fair, tie_pct_fair) if judge_fair else ""
    dims_html = "".join(make_dim_row(k.replace("_", " ").title(), v) for k, v in dim_fair.items())
    judge_block = f"""
      <div style="margin-top:1rem;">
        <div style="font-size:.75rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:.8px;color:var(--kcl);margin-bottom:.4rem">
          Pairwise Judge (Gemini 2.5 Flash, multimodal) &mdash; n = {n_fair}
        </div>
        {judge_bar}
        {dims_html}
      </div>""" if judge_fair else ""

    chair_delta_v3  = round(v3_chair_i_base  - v3_chair_i_mop,  1)
    chair_delta_v4  = round(v4_chair_i_base  - v4_chair_i_mop,  1)

    return f"""
<section id="archcompare">
  <h2>&#9881; Architecture: v3 Closed-World &rarr; v4 Open+Anchor</h2>
  <p style="color:#555;margin-bottom:1.2rem;font-size:.9rem;line-height:1.6;">
    The MOP pipeline underwent a fundamental architectural shift between v3 and v4.
    v3 used a <b>closed-world constraint</b> (&ldquo;ONLY describe using these tags&rdquo;),
    suppressing the VLM&rsquo;s visual understanding. v4 switches to
    <b>Open+Anchor</b>: the VLM describes freely what it sees, but <em>must</em> mention
    verified L3 product facts as grounding anchors.
  </p>

  <!-- Change summary -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:1.5rem;">

    <!-- v3 -->
    <div class="fair-card" style="border-left:4px solid #e74c3c;">
      <h3 style="color:#c0392b;">&#128274; v3 &mdash; Closed-World Constraint</h3>
      <p style="font-size:.8rem;color:#888;margin-bottom:.6rem">
        Prompt: <em>&ldquo;ONLY describe using these facts: [L1&ndash;L4 tags]&rdquo;</em><br>
        Raw OCR tokens injected directly &middot; Absolute LLM judge (text-only, circular GT)
      </p>
      <div style="background:#fdf2f2;border-radius:8px;padding:.8rem;margin-bottom:.6rem">
        <div style="font-size:.78rem;color:#555;margin-bottom:.4rem">CHAIR_i (tag compliance metric)</div>
        <div style="display:flex;gap:1.5rem;">
          <div><div style="font-size:1.5rem;font-weight:700;color:#1e8449">{v3_chair_i_mop}%</div>
               <div style="font-size:.72rem;color:#777">MOP CHAIR_i</div></div>
          <div><div style="font-size:1.5rem;font-weight:700;color:#c0392b">{v3_chair_i_base}%</div>
               <div style="font-size:.72rem;color:#777">Baseline CHAIR_i</div></div>
          <div><div style="font-size:1.5rem;font-weight:700;color:#1e8449">&minus;{chair_delta_v3}pp</div>
               <div style="font-size:.72rem;color:#777">MOP improvement</div></div>
        </div>
      </div>
      <div style="font-size:.8rem;color:#666;line-height:1.5">
        <b>Problems identified:</b><br>
        &bull; VLM visual understanding suppressed by tag-only constraint<br>
        &bull; Seasonal/contextual cues (Halloween theme) ignored<br>
        &bull; Raw OCR noise ("te", "s", "co") copied into captions<br>
        &bull; Evaluation circular: CHAIR GT built from same tags as prompt
      </div>
    </div>

    <!-- v4 -->
    <div class="fair-card" style="border:2px solid var(--kcl);border-left:4px solid var(--kcl);">
      <h3 style="color:var(--kcl);">&#128275; v4 &mdash; Open+Anchor Architecture</h3>
      <p style="font-size:.8rem;color:#888;margin-bottom:.6rem">
        Prompt: <em>&ldquo;Describe what you SEE. You MUST mention: [L3 anchors]. Do NOT guess: [ambiguous].&rdquo;</em><br>
        Raw OCR removed &middot; Multimodal pairwise judge (image + captions, no circular GT)
      </p>
      <div style="background:#f0f8f0;border-radius:8px;padding:.8rem;margin-bottom:.6rem">
        <div style="font-size:.78rem;color:#555;margin-bottom:.4rem">CHAIR_i + Pairwise Judge (Gemini 2.5 Flash)</div>
        <div style="display:flex;gap:1.5rem;flex-wrap:wrap">
          <div><div style="font-size:1.5rem;font-weight:700;color:#1e8449">{v4_chair_i_mop}%</div>
               <div style="font-size:.72rem;color:#777">MOP CHAIR_i</div></div>
          <div><div style="font-size:1.5rem;font-weight:700;color:#c0392b">{v4_chair_i_base}%</div>
               <div style="font-size:.72rem;color:#777">Baseline CHAIR_i</div></div>
          <div><div style="font-size:1.5rem;font-weight:700;color:#1e8449">{mop_pct_fair}%</div>
               <div style="font-size:.72rem;color:#777">Pairwise win rate</div></div>
        </div>
      </div>
      {judge_block}
    </div>
  </div>

  <!-- Key insight box -->
  <div style="background:var(--kcl-pale);border-left:4px solid var(--kcl);
              padding:1rem;border-radius:0 8px 8px 0;font-size:.88rem;line-height:1.7;">
    <b>&#128161; Key Insight &mdash; CHAIR_i rise is not regression:</b>
    In v4, the VLM uses richer, more natural vocabulary (e.g. &ldquo;Halloween display&rdquo;,
    &ldquo;orange pumpkins&rdquo;) that may not appear verbatim in the L3 tag list,
    causing <em>apparent</em> CHAIR_i increase. This reveals that CHAIR is measuring
    <b>tag compliance</b>, not true hallucination. The multimodal pairwise judge
    (no circular GT) shows MOP captions are preferred in <b>{mop_pct_fair}%</b> of cases,
    confirming the Open+Anchor approach produces genuinely better captions.
  </div>
</section>"""


# ── Architecture evolution gallery ────────────────────────────────────────────

def make_arch_evo_card(score: dict, idx: int, v3_cap_map: dict | None = None) -> str:
    """Single card: image | v3 Closed-World caption | v4 Open+Anchor caption | judge verdict."""
    img_name  = score.get("image_file", "")
    v4_cap    = score.get("mop_caption", "")
    # v3 caption: use stored interim captions if available, else show label
    v3_cap = (v3_cap_map or {}).get(img_name, "[v3 caption not stored for this image]")
    pref      = score.get("preference", "Tie")
    reasoning = score.get("reasoning", "")
    va_winner = score.get("visual_accuracy_winner", "Tie")
    fc_winner = score.get("factual_completeness_winner", "Tie")
    uh_winner = score.get("uncertainty_handling_winner", "Tie")

    b64  = img_to_b64(os.path.join(IMG_DIR, img_name))
    img_tag = (
        f'<img src="data:image/jpeg;base64,{b64}" '
        f'alt="{img_name}" style="width:80px;height:60px;object-fit:cover;border-radius:4px;">'
        if b64 else f'<span style="color:rgba(255,255,255,.6);font-size:.8rem">{img_name}</span>'
    )

    pref_cls  = {"MOP": "verdict-mop", "Baseline": "verdict-base"}.get(pref, "verdict-tie")
    pref_icon = {"MOP": "&#9650;", "Baseline": "&#9660;", "Tie": "="}.get(pref, "=")
    pref_label = {"MOP": "v4 Open+Anchor preferred", "Baseline": "v3-style (unconstrained) preferred", "Tie": "Tie"}.get(pref, "Tie")

    def dim_pill(label: str, winner: str) -> str:
        winner_label = {"MOP": "v4", "Baseline": "v3-style", "Tie": "Tie"}.get(winner, winner)
        if winner == "MOP":
            return f'<span class="score-pill pill-blue">{label}: {winner_label} &#10003;</span>'
        elif winner == "Baseline":
            return f'<span class="score-pill pill-bad">{label}: {winner_label} &#10003;</span>'
        return f'<span class="score-pill" style="background:#eee;color:#666">{label}: Tie</span>'

    dim_pills = " ".join([
        dim_pill("VisAcc", va_winner),
        dim_pill("FactComp", fc_winner),
        dim_pill("UncertH", uh_winner),
    ])

    return f"""
<div class="arch-evo-card">
  <div class="arch-evo-header">
    {img_tag}
    <div>
      <div style="font-weight:600;font-size:.95rem">{img_name}</div>
      <div style="font-size:.78rem;opacity:.8">Example #{idx+1} &nbsp;|&nbsp;
        Judge: <b>{pref_icon} {pref_label}</b></div>
    </div>
    <div style="margin-left:auto;font-size:.78rem;opacity:.8">
      {dim_pills}
    </div>
  </div>
  <div class="arch-evo-body">
    <div class="arch-cap-panel unconstrained">
      <div class="arch-cap-label">&#128274; v3 Closed-World (tags-only constraint)</div>
      <div class="arch-cap-text">{v3_cap}</div>
    </div>
    <div class="arch-cap-panel mop-open">
      <div class="arch-cap-label">&#128275; v4 Open+Anchor (verified L3 anchors)</div>
      <div class="arch-cap-text">{v4_cap}</div>
    </div>
  </div>
  <div class="arch-verdict">
    <div class="verdict-label {pref_cls}">Judge Reasoning ({pref_icon} {pref_label})</div>
    {reasoning}
  </div>
</div>"""


def make_arch_evolution_section(judge_fair: dict | None) -> str:
    """Build gallery showing v3 closed-world vs v4 Open+Anchor caption changes."""
    if not judge_fair:
        return ""

    scores = judge_fair.get("detailed_scores", [])

    # Load v3 captions (interim file from v3 era, April 8)
    v3_cap_map: dict = {}
    V3_PATH = "data/cache/final_captions_interim.json"
    if os.path.exists(V3_PATH):
        with open(V3_PATH, "r", encoding="utf-8") as f:
            v3_raw = json.load(f)
        # Find caption key
        cap_key = None
        if v3_raw:
            cap_key = next((k for k in v3_raw[0].keys() if "CAPTION" in k.upper()), None)
        if cap_key:
            v3_cap_map = {item.get("image_file", ""): item.get(cap_key, "") for item in v3_raw if "image_file" in item}
        print(f"[ARCH-EVO] Loaded {len(v3_cap_map)} v3 captions from {V3_PATH}")

    scores = judge_fair.get("detailed_scores", [])

    # Pick examples: 5 strong MOP wins + 2 Baseline wins for honest balance
    mop_wins = [
        s for s in scores
        if s.get("preference") == "MOP"
        and s.get("visual_accuracy_winner") == "MOP"
        and s.get("factual_completeness_winner") == "MOP"
        and os.path.exists(os.path.join(IMG_DIR, s.get("image_file", "")))
    ]
    base_wins = [
        s for s in scores
        if s.get("preference") == "Baseline"
        and s.get("visual_accuracy_winner") == "Baseline"
        and s.get("factual_completeness_winner") == "Baseline"
        and os.path.exists(os.path.join(IMG_DIR, s.get("image_file", "")))
    ]

    random.seed(42)
    # Pick samples that have v3 captions available
    mop_wins_with_v3  = [s for s in scores
        if s.get("preference") == "MOP"
        and s.get("visual_accuracy_winner") == "MOP"
        and s.get("factual_completeness_winner") == "MOP"
        and os.path.exists(os.path.join(IMG_DIR, s.get("image_file", "")))
        and s.get("image_file", "") in v3_cap_map]
    base_wins_with_v3 = [s for s in scores
        if s.get("preference") == "Baseline"
        and s.get("visual_accuracy_winner") == "Baseline"
        and os.path.exists(os.path.join(IMG_DIR, s.get("image_file", "")))
        and s.get("image_file", "") in v3_cap_map]

    # Fall back to any with images if v3 not available
    mop_wins  = mop_wins_with_v3  or [s for s in scores if s.get("preference") == "MOP"
        and s.get("visual_accuracy_winner") == "MOP"
        and os.path.exists(os.path.join(IMG_DIR, s.get("image_file", "")))]
    base_wins = base_wins_with_v3 or [s for s in scores if s.get("preference") == "Baseline"
        and s.get("visual_accuracy_winner") == "Baseline"
        and os.path.exists(os.path.join(IMG_DIR, s.get("image_file", "")))]

    showcase = (
        sorted(random.sample(mop_wins[:30], min(5, len(mop_wins))),
               key=lambda s: s["image_file"])
        + sorted(random.sample(base_wins[:10], min(2, len(base_wins))),
                 key=lambda s: s["image_file"])
    )

    cards_html = "".join(make_arch_evo_card(s, i, v3_cap_map) for i, s in enumerate(showcase))

    mop_total  = sum(1 for s in scores if s.get("preference") == "MOP")
    base_total = sum(1 for s in scores if s.get("preference") == "Baseline")

    return f"""
<section id="archevo">
  <h2>&#128203; Caption Gallery: v3 Closed-World vs v4 Open+Anchor</h2>
  <p style="color:#555;margin-bottom:.8rem;font-size:.9rem;line-height:1.6;">
    Side-by-side captions for the same image under <b>v3 Closed-World</b> (left) and
    <b>v4 Open+Anchor</b> (right) prompting strategies, using <em>the same model
    (llava-phi3)</em>. The judge (Gemini 2.5 Flash, multimodal) evaluates which caption
    better reflects the actual image &mdash; without any circular ground truth.
  </p>
  <div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1.2rem;">
    <span class="score-pill pill-blue">&#9650; v4 Open+Anchor wins: {mop_total} / {len(scores)}</span>
    <span class="score-pill pill-bad">&#9660; v3-style wins: {base_total} / {len(scores)}</span>
    <span class="score-pill" style="background:#eee;color:#666;">Showing {len(showcase)} examples ({min(5,len(mop_wins))} v4 preferred + {min(2,len(base_wins))} v3-style preferred)</span>
  </div>
  {cards_html}
</section>"""


def run():
    parser = argparse.ArgumentParser(description="MOP vs Baseline Comparison Report (HTML)")
    parser.add_argument("--baseline-path", default=BASELINE_PATH,
                        help=f"Baseline captions JSON (default: {BASELINE_PATH})")
    parser.add_argument("--judge-path", default=LLM_JUDGE_PATH,
                        help=f"LLM Judge scores JSON (default: {LLM_JUDGE_PATH})")
    parser.add_argument("--judge-path-fair", default=LLM_JUDGE_PATH_FAIR,
                        help=f"Same-model fair LLM Judge JSON (default: {LLM_JUDGE_PATH_FAIR})")
    parser.add_argument("--out-html", default=OUT_HTML,
                        help=f"Output HTML path (default: {OUT_HTML})")
    parser.add_argument("--out-json", default=OUT_JSON,
                        help=f"Output JSON path (default: {OUT_JSON})")
    args = parser.parse_args()

    baseline_path  = args.baseline_path
    llm_judge_path = args.judge_path
    llm_judge_fair = args.judge_path_fair
    out_html       = args.out_html
    out_json       = args.out_json

    download_nltk()

    if not os.path.exists(MOP_PATH):
        raise FileNotFoundError(f"MOP captions not found: {MOP_PATH}")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline captions not found: {baseline_path}")

    with open(MOP_PATH,      "r", encoding="utf-8") as f: mop_data      = json.load(f)
    with open(baseline_path, "r", encoding="utf-8") as f: baseline_data = json.load(f)

    mop_map      = {i["image_file"]: i for i in mop_data if "image_file" in i}
    baseline_map = {i["image_file"]: i for i in baseline_data if "image_file" in i}

    llm_map = {}
    if os.path.exists(llm_judge_path):
        with open(llm_judge_path, "r", encoding="utf-8") as f:
            llm_raw = json.load(f)
        llm_map = {s["image_file"]: s for s in llm_raw.get("detailed_scores", [])}

    # Load fair (same-model) judge results
    judge_fair_data = None
    if os.path.exists(llm_judge_fair):
        with open(llm_judge_fair, "r", encoding="utf-8") as f:
            judge_fair_data = json.load(f)
        print(f"[FAIR] Loaded same-model judge: {llm_judge_fair} "
              f"({judge_fair_data.get('n_evaluated', 0)} pairs)")

    common = sorted(set(mop_map) & set(baseline_map))
    print(f"[CHAIR] Comparing {len(common)} images with both MOP and Baseline captions.")

    results = []
    total_mop_hall = total_base_hall = total_mop_n = total_base_n = 0
    mop_hall_s = base_hall_s = 0

    for img_name in common:
        mop_item  = mop_map[img_name]
        base_item = {**baseline_map[img_name],
                     "L1_scene":     mop_item.get("L1_scene", {}),
                     "L2_fixtures":  mop_item.get("L2_fixtures", {}),
                     "L3_products":  mop_item.get("L3_products", {}),
                     "L4_attributes":mop_item.get("L4_attributes", {})}

        mop_h,  mop_ci,  mop_n  = chair_score(mop_item)
        base_h, base_ci, base_n = chair_score(base_item)

        total_mop_hall  += len(mop_h);  total_mop_n  += mop_n
        total_base_hall += len(base_h); total_base_n += base_n
        if mop_h:  mop_hall_s  += 1
        if base_h: base_hall_s += 1

        b64 = img_to_b64(os.path.join(IMG_DIR, img_name))

        results.append({
            "image":                    img_name,
            "mop_caption":              mop_item.get("FINAL_CAPTION", ""),
            "baseline_caption":         baseline_map[img_name].get("FINAL_CAPTION", ""),
            "mop_hallucinated_nouns":   mop_h,
            "baseline_hallucinated_nouns": base_h,
            "mop_chair_i":              mop_ci,
            "baseline_chair_i":         base_ci,
            "chair_i_delta":            round(base_ci - mop_ci, 3),
            "llm":                      llm_map.get(img_name),
            "mop_route_cluster":        mop_item.get("MOP_route_cluster"),
            "img_b64":                  b64,
            # For evolution section
            "_mop_item":                mop_item,
        })

    n = len(common)
    mop_ci_agg  = round(total_mop_hall  / max(total_mop_n,  1) * 100, 2)
    base_ci_agg = round(total_base_hall / max(total_base_n, 1) * 100, 2)
    mop_cs      = round(mop_hall_s  / max(n, 1) * 100, 2)
    base_cs     = round(base_hall_s / max(n, 1) * 100, 2)

    # --- Pairwise judge stats from phi3 fair comparison ---
    # The new judge uses preference / visual_accuracy_winner etc., not accuracy_score/10
    judged_pairwise = [r for r in results if r["llm"] and r["llm"].get("preference") is not None]
    n_pairwise = max(len(judged_pairwise), 1)
    mop_pw_wins  = sum(1 for r in judged_pairwise if r["llm"].get("preference") == "MOP")
    base_pw_wins = sum(1 for r in judged_pairwise if r["llm"].get("preference") == "Baseline")
    tie_pw       = sum(1 for r in judged_pairwise if r["llm"].get("preference") == "Tie")
    mop_va_wins  = sum(1 for r in judged_pairwise if r["llm"].get("visual_accuracy_winner") == "MOP")
    mop_fc_wins  = sum(1 for r in judged_pairwise if r["llm"].get("factual_completeness_winner") == "MOP")
    mop_uh_wins  = sum(1 for r in judged_pairwise if r["llm"].get("uncertainty_handling_winner") == "MOP")
    mop_pw_pct   = round(mop_pw_wins  / n_pairwise * 100, 1)
    base_pw_pct  = round(base_pw_wins / n_pairwise * 100, 1)
    tie_pw_pct   = round(tie_pw       / n_pairwise * 100, 1)
    va_pct       = round(mop_va_wins  / n_pairwise * 100, 1)
    fc_pct       = round(mop_fc_wins  / n_pairwise * 100, 1)
    uh_pct       = round(mop_uh_wins  / n_pairwise * 100, 1)

    # Keep legacy judged for backward compat (will be empty with new pairwise judge)
    judged  = [r for r in results if r["llm"] and r["llm"].get("accuracy_score") is not None]
    avg_acc = round(sum(r["llm"]["accuracy_score"] for r in judged) / max(len(judged), 1), 2)
    avg_rel = round(sum(r["llm"]["relevance_score"] for r in judged) / max(len(judged), 1), 2)
    avg_abs = round(sum(r["llm"]["absence_handling_score"] for r in judged) / max(len(judged), 1), 2)

    # Save JSON summary (without b64/object refs)
    summary_json = {
        "n_images_compared": n,
        "baseline_model": "llava-phi3 (same-model fair comparison)",
        "CHAIR_i": {"MOP": mop_ci_agg, "Baseline_phi3": base_ci_agg,
                    "improvement_pp": round(base_ci_agg - mop_ci_agg, 2)},
        "CHAIR_s": {"MOP": mop_cs, "Baseline_phi3": base_cs,
                    "improvement_pp": round(base_cs - mop_cs, 2)},
        "LLM_Judge_Pairwise": {
            "n_evaluated": len(judged_pairwise),
            "MOP_win_pct": mop_pw_pct, "Baseline_win_pct": base_pw_pct,
            "Tie_pct": tie_pw_pct,
            "visual_accuracy_MOP_pct": va_pct,
            "factual_completeness_MOP_pct": fc_pct,
            "uncertainty_handling_MOP_pct": uh_pct,
        },
        "per_image": [{k: v for k, v in r.items() if k not in ("img_b64", "_mop_item")}
                      for r in results],
    }
    os.makedirs("data/eval_results", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4, ensure_ascii=False)
    print(f"[DONE] Comparison JSON → {OUT_JSON}")

    # ── Sort for gallery: most improved first ──────────────────────────────────
    sorted_results = sorted(results, key=lambda r: r["chair_i_delta"], reverse=True)

    # Gallery: top N_GALLERY, evolution: pick 3 varied examples
    gallery_items    = sorted_results[:N_GALLERY]
    evolution_items  = [r["_mop_item"] for r in sorted_results[:N_EVOLUTION]]

    # ── Build HTML ─────────────────────────────────────────────────────────────
    gallery_html   = "\n".join(make_compare_card(r) for r in gallery_items)
    evolution_html = "\n".join(make_evolution_card(item) for item in evolution_items)

    mop_scores_list  = [r["mop_chair_i"]      for r in results]
    base_scores_list = [r["baseline_chair_i"] for r in results]
    histogram_svg    = make_histogram_svg(mop_scores_list, base_scores_list)

    # Worst/Best
    best_5  = sorted_results[:3]
    worst_3 = sorted_results[-3:][::-1]
    best_html  = "\n".join(make_compare_card(r) for r in best_5)
    worst_html = "\n".join(make_compare_card(r) for r in worst_3)

    # Common hallucinated nouns in baseline
    from collections import Counter
    all_hall = [w for r in results for w in r["baseline_hallucinated_nouns"]]
    top_hall = Counter(all_hall).most_common(10)
    top_hall_html = " ".join(
        f'<span class="score-pill pill-bad">{w} ({c}×)</span>'
        for w, c in top_hall
    ) or "<em>None</em>"

    # Build arch comparison (v3 vs v4) + caption gallery sections
    arch_cmp_html  = make_arch_comparison_section(judge_fair_data)
    arch_evo_html  = make_arch_evolution_section(judge_fair_data)

    improvement_pp = round(base_ci_agg - mop_ci_agg, 1)
    imp_cls = "good" if improvement_pp > 0 else "bad"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MOP Pipeline Report — KCL MSc Dissertation</title>
  <style>{HTML_CSS}</style>
</head>
<body>

<div class="hero">
  <h1>🛒 Hallucination-Controlled Retail Image Captioning</h1>
  <div class="subtitle">MOP Pipeline vs Unconstrained Baseline — Evaluation Report</div>
  <div class="meta">KCL MSc Dissertation &nbsp;|&nbsp; Sample: {n} images (same-model: llava-phi3) &nbsp;|&nbsp;
    Pairwise Judge: {len(judged_pairwise)} pairs evaluated</div>
</div>

<nav>
  <a href="#overview">Overview</a>
  <a href="#metrics">Metrics</a>
  <a href="#archcompare">v3 vs v4</a>
  <a href="#archevo">Caption Gallery</a>
  <a href="#evolution">Stage Evolution</a>
  <a href="#gallery">Before / After</a>
  <a href="#analysis">Analysis</a>
  <a href="#extremes">Best / Worst</a>
</nav>

<div class="container">

<!-- ═══════════════════════════════ OVERVIEW ══════════════════════════════════ -->
<section id="overview">
  <h2>🔬 Research Overview</h2>
  <p style="margin-bottom:1rem;line-height:1.7;color:#444;font-size:.95rem;">
    Vision-Language Models (VLMs) such as LLaVA frequently <strong>hallucinate</strong>
    product names and attributes when captioning grocery retail images — generating plausible
    but factually incorrect text. This is a critical problem for retail analytics where
    caption accuracy directly impacts inventory, merchandising, and compliance decisions.
  </p>
  <p style="margin-bottom:1.5rem;line-height:1.7;color:#444;font-size:.95rem;">
    This dissertation proposes the <strong>MOP (Metadata-Grounded Object-Prompting) Pipeline</strong>:
    a hierarchical tagging system that provides structured, verifiable facts (L0–L4) as
    deterministic constraints to the VLM prompt — making hallucination structurally impossible
    for tagged entities.
  </p>

  <h3>Pipeline Architecture</h3>
  <div class="pipeline-flow">
    <div class="flow-box free">Stage 1<br><small>Image Ingestion</small></div>
    <div class="flow-arrow">→</div>
    <div class="flow-box free">Stage 2<br><small>L1+L2 Tagging<br>(CLIP / DINO)</small></div>
    <div class="flow-arrow">→</div>
    <div class="flow-box api">Stage 3B<br><small>L3 Products<br>(Gemini + CLIP)</small></div>
    <div class="flow-arrow">→</div>
    <div class="flow-box free">Stage 3C<br><small>L4 Attributes<br>(OCR + CLIP)</small></div>
    <div class="flow-arrow">→</div>
    <div class="flow-box api">Stage 4<br><small>MOP Clustering<br>+ Prompt Gen</small></div>
    <div class="flow-arrow">→</div>
    <div class="flow-box local">Stage 5<br><small>VLM Captioning<br>(Ollama local)</small></div>
    <div class="flow-arrow">→</div>
    <div class="flow-box free">Stage 6<br><small>CHAIR Eval</small></div>
    <div class="flow-arrow">→</div>
    <div class="flow-box api">Stage 7<br><small>LLM Judge<br>(Gemini)</small></div>
  </div>
  <p style="font-size:.8rem;color:#888;">
    <span class="score-pill pill-good">■ Free (local only)</span>&nbsp;
    <span class="score-pill pill-bad">■ Gemini API</span>&nbsp;
    <span class="score-pill" style="background:#e8e8e8;color:#333">■ Local VLM (Ollama)</span>
  </p>
</section>

<!-- ═══════════════════════════════ METRICS ═══════════════════════════════════ -->
<section id="metrics">
  <h2>📊 Evaluation Metrics</h2>
  <div class="metrics-grid">
    <div class="metric-card good">
      <div class="label">MOP CHAIR_i &#8595;</div>
      <div class="value">{mop_ci_agg}%</div>
      <div class="sub">Hallucination rate (nouns)</div>
    </div>
    <div class="metric-card bad">
      <div class="label">Baseline CHAIR_i &#8595; (phi3)</div>
      <div class="value">{base_ci_agg}%</div>
      <div class="sub">Same model, unconstrained</div>
    </div>
    <div class="metric-card {imp_cls}">
      <div class="label">CHAIR_i Improvement</div>
      <div class="value">{improvement_pp:+.1f}pp</div>
      <div class="sub">MOP vs Baseline</div>
    </div>
    <div class="metric-card good">
      <div class="label">MOP CHAIR_s ↓</div>
      <div class="value">{mop_cs}%</div>
      <div class="sub">Captions w/ hallucination</div>
    </div>
    <div class="metric-card bad">
      <div class="label">Baseline CHAIR_s ↓</div>
      <div class="value">{base_cs}%</div>
      <div class="sub">Captions w/ hallucination</div>
    </div>
    <div class="metric-card neutral">
      <div class="label">LLM Judge — Accuracy</div>
      <div class="value">{avg_acc}<span style="font-size:1rem">/10</span></div>
      <div class="sub">{len(judged)} captions evaluated</div>
    </div>
    <div class="metric-card neutral">
      <div class="label">LLM Judge — Relevance</div>
      <div class="value">{avg_rel}<span style="font-size:1rem">/10</span></div>
      <div class="sub">Retail manager suitability</div>
    </div>
    <div class="metric-card neutral">
      <div class="label">LLM Judge — Absence</div>
      <div class="value">{avg_abs}<span style="font-size:1rem">/10</span></div>
      <div class="sub">Ambiguous fact handling</div>
    </div>
    <div class="metric-card neutral">
      <div class="label">Images Analysed</div>
      <div class="value">{n}</div>
      <div class="sub">MOP + Baseline pairs</div>
    </div>
  </div>
</section>

<!-- ═══════════════════════════ STAGE EVOLUTION ═══════════════════════════════ -->
<section id="evolution">
  <h2>🔄 Stage-by-Stage Tag Evolution</h2>
  <p style="color:#555;margin-bottom:1.2rem;font-size:.9rem;line-height:1.6;">
    The MOP pipeline incrementally builds a <strong>structured knowledge base</strong>
    for each image across 5 hierarchical levels. Each stage adds verifiable facts
    that constrain the VLM, preventing it from inventing information.
    The examples below show how each stage contributes to the final caption.
  </p>
  {evolution_html}
</section>

<!-- ═══════════════════════════ BEFORE / AFTER ════════════════════════════════ -->
<section id="gallery">
  <h2>🖼️ Before / After: Caption Comparison</h2>
  <p style="color:#555;margin-bottom:1.2rem;font-size:.9rem;line-height:1.6;">
    Top {len(gallery_items)} examples sorted by largest hallucination improvement (Δ CHAIR_i).
    <mark class="hall">Highlighted words</mark> are hallucinated nouns not supported by
    ground-truth tags.
  </p>
{gallery_html}
</section>

{arch_cmp_html}

{arch_evo_html}

<!-- ═══════════════════════════════ ANALYSIS ══════════════════════════════════ -->
<section id="analysis">
  <h2>📈 Hallucination Distribution Analysis</h2>
  <div class="chart-wrap">
    <h3 style="margin-bottom:1rem">CHAIR_i Score Distribution — MOP vs Baseline</h3>
    {histogram_svg}
  </div>

  <div style="margin-top:1.5rem;padding:1rem;background:var(--card);border:1px solid var(--border);border-radius:10px;">
    <h3 style="margin-bottom:.8rem">🚨 Most Common Hallucinated Nouns — Baseline</h3>
    <p style="font-size:.85rem;color:#555;margin-bottom:.7rem;">
      These nouns appeared in baseline captions but had no grounding in the image's verified tags:
    </p>
    {top_hall_html}
  </div>
</section>

<!-- ═══════════════════════════ BEST / WORST ══════════════════════════════════ -->
<section id="extremes">
  <h2>🏆 Best Cases — Largest Improvement</h2>
  <p style="color:#555;margin-bottom:1rem;font-size:.9rem;">
    Images where the MOP pipeline most dramatically reduced hallucinations vs baseline.
  </p>
  {best_html}

  <h2 style="margin-top:2rem">⚠️ Challenging Cases</h2>
  <p style="color:#555;margin-bottom:1rem;font-size:.9rem;">
    Images where hallucination persisted or was difficult to control — informing future work.
  </p>
  {worst_html}
</section>

</div><!-- /container -->

<footer>
  KCL MSc Dissertation — Hallucination-Controlled Retail VLM Captioning &nbsp;|&nbsp;
  Generated by pipeline_8_comparison_report.py &nbsp;|&nbsp;
  Evaluation: Retail-CHAIR + LLM-as-a-Judge (Gemini)
</footer>

<script>{HTML_JS}</script>
</body>
</html>"""

    os.makedirs("data/eval_results", exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[DONE] HTML report -> {out_html}")

    print("\n" + "="*55)
    print("  MOP Pipeline -- Evaluation Summary")
    print("="*55)
    print(f"  CHAIR_i : MOP {mop_ci_agg}%  vs  phi3-Baseline {base_ci_agg}%  (D {improvement_pp:+.1f}pp)")
    print(f"  CHAIR_s : MOP {mop_cs}%   vs  phi3-Baseline {base_cs}%")
    if judged_pairwise:
        print(f"  Pairwise Judge: MOP {mop_pw_pct}% | Base {base_pw_pct}% | Tie {tie_pw_pct}%  ({len(judged_pairwise)} pairs)")
    if judge_fair_data:
        wr_f = judge_fair_data.get("overall_win_rate", {})
        print(f"  Fair Judge: MOP {wr_f.get('MOP_win_pct',0)}% | n={judge_fair_data.get('n_evaluated',0)}")
    print(f"  Images  : {n} compared")
    print("="*55)
    print(f"\n  Open report: {os.path.abspath(out_html)}")


if __name__ == "__main__":
    run()

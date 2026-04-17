# Tag-Constrained Retail VLM Captioning

> **KCL MSc Dissertation** — Hallucination-Controlled Retail Image Captioning  
> **Version**: v4 — Open+Anchor Architecture  
> **Last Updated**: 2026-04-17

---

## Research Overview

Vision-Language Models (VLMs) hallucinate product names and attributes when captioning grocery retail images — a critical problem for retail analytics where caption accuracy directly impacts inventory and merchandising decisions.

This project proposes the **MOP (Metadata-Grounded Object-Prompting) Pipeline**: a hierarchical tagging system that provides structured, verifiable facts as anchors to a lightweight local VLM, constraining its output without suppressing its visual understanding.

**Key design principle (v4)**:  
> Tags should *augment* the VLM's visual understanding, not *replace* it.  
> The VLM describes what it sees freely; verified L3 product facts are mandatory anchors.

---

## Pipeline Architecture

```
[HuggingFace Dataset]
        │
        ▼
Stage 1: Image Ingestion & Quality Filter
  pipeline_1_ingestion.py
        │
        ▼
Stage 2: GS1 Ontology Mapping (one-time)
  pipeline_2_ontology.py
        │
        ▼
Stage 3A: L1 Scene + L2 Fixture Tagging (CLIP + GroundingDINO)
  pipeline_3_dynamic_tagging.py
        │
        ▼
Stage 3B: L3 Product Tagging (Gemini Vision + CLIP, API)
  pipeline_3b_l3_product_tagging.py
        │
        ▼
Stage 3C: L4 Attribute Tagging (OCR + CLIP, local)
  pipeline_3c_l4_attribute_tagging.py
        │
        ▼
Stage 4: MOP Routing + Prompt Generation (K-Means + Gemini, API)
  pipeline_4_routing_clustering.py
        │
        ▼
Stage 5: VLM Captioning — Open+Anchor MOP Prompt (Ollama, local)
  pipeline_5_mop_captioning.py        [UPDATED v4]
        │
        ├──────────────────────────────────────────────────┐
        │                                                  │
Stage 6: Retail-CHAIR Evaluation               Baseline: Unconstrained VLM
  pipeline_6_eval_chair.py             pipeline_ablation_baseline.py [UPDATED v4]
        │
        ▼
Stage 7: Multimodal Pairwise LLM Judge (Gemini 2.5 Flash, API)
  pipeline_7_llm_judge.py             [UPDATED v4 — Pairwise, Multimodal]
        │
        ▼
Stage 8: HTML Comparison Report
  pipeline_8_comparison_report.py     [UPDATED v4 — Full HTML rewrite]
```

---

## Hierarchical Tag Layers

| Level | Content | Method |
|---|---|---|
| **L0** | Global context (retailer, season, campaign) | Metadata |
| **L1** | Scene classification (aisle / endcap / till / display bin) | CLIP zero-shot |
| **L2** | Fixture detection (shelf, display box, bin…) | GroundingDINO |
| **L3** | Product tagging (Hard / Soft / Absence) | Gemini Vision + CLIP |
| **L4** | Operational state (stock, tidiness, promotion) + OCR | CLIP + PaddleOCR |

---

## v4 Key Changes (2026-04-17)

### Problem Identified (100-sample experiment)
The v3 pipeline used a **closed-world constraint** prompting strategy that *replaced* the VLM's visual reasoning with tag labels. This caused:
- Loss of visual context (seasonal themes, colors, layout not captured)
- Garbled captions due to raw OCR noise injection
- Circular evaluation: CHAIR GT built from the same tags used for generation

### Solution: Open+Anchor Architecture
```
# v3 (Closed-world)
"Only describe using these facts: [L1-L4 tags]"

# v4 (Open+Anchor)
"Describe what you SEE. You MUST include: [L3 verified anchors].
 Do NOT guess: [Ambiguous L4 attributes]."
```

### Evaluation Upgrade: Multimodal Pairwise Judge
- v3: Absolute scores, text-only → circular logic
- v4: Image + both captions → Gemini judge → pairwise win-rate  
  (3 dimensions: visual accuracy / factual completeness / uncertainty handling)

See [CHANGELOG.md](./CHANGELOG.md) for full details.

---

## Quick Start

```bash
# 1. Setup environment
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 2. Run stages sequentially (or selectively)
run_pipeline.bat

# 3. Run a 100-sample experiment
python src/pipeline_5_mop_captioning.py --sample 100 --model llava-phi3
python src/pipeline_ablation_baseline.py --from-mop
python src/pipeline_6_eval_chair.py
python src/pipeline_7_llm_judge.py --sample 30
python src/pipeline_8_comparison_report.py
# → data/eval_results/comparison_report.html
```

### Prerequisites
- **Ollama** running locally with `llava-phi3` or `llava` installed
- **Gemini API key** in `.env`: `GEMINI_API_KEY=your_key`
- Python 3.11+, CUDA GPU recommended

---

## Known Limitations (v4 experiment)

1. **Model size mismatch**: MOP uses llava-phi3 (3.8B), Baseline uses llava (7B).  
   Fair comparison requires same base model — planned for next experiment.

2. **L3 tag quality unvalidated**: No independent verification that L3 product tags match ground truth. Tag-level hallucination is undetected by current evaluation.

3. **CHAIR circular evaluation**: CHAIR GT is derived from the same L1-L4 tags used to constrain generation. CHAIR measures tag compliance, not true hallucination. Multimodal pairwise judge is now the primary metric.

4. **Two-pass architecture not implemented**: A RAG-style Pass1 (free generation) + Pass2 (tag-based post-hoc grounding) is a stronger research contribution — planned for next iteration.

---

## Evaluation Results (100-sample, v4)

| Metric | MOP Pipeline | Baseline |
|---|---|---|
| CHAIR_i | 78.41% | 89.98% |
| Pairwise Win Rate | 40.7% | 55.6% |
| Ties | — | 3.7% |

*Note: CHAIR_i increase vs v3 is expected and intended — it demonstrates CHAIR's circular evaluation limitation. Pairwise results are influenced by model size mismatch.*

---

## Project Structure

```
antigravity/
├── src/
│   ├── pipeline_1_ingestion.py
│   ├── pipeline_2_ontology.py
│   ├── pipeline_2b_gs1_mapper.py
│   ├── pipeline_3_dynamic_tagging.py
│   ├── pipeline_3b_l3_product_tagging.py
│   ├── pipeline_3c_l4_attribute_tagging.py
│   ├── pipeline_4_routing_clustering.py    # Prompt template generation
│   ├── pipeline_5_mop_captioning.py        # [v4] Open+Anchor VLM captioning
│   ├── pipeline_6_eval_chair.py            # Retail-CHAIR evaluation
│   ├── pipeline_7_llm_judge.py             # [v4] Multimodal Pairwise Judge
│   ├── pipeline_8_comparison_report.py     # [v4] HTML report generation
│   └── pipeline_ablation_baseline.py       # [v4] Baseline with --from-mop
├── data/
│   ├── processed/          # Downloaded images
│   ├── cache/              # Intermediate tag/caption JSON files
│   └── eval_results/       # CHAIR metrics, judge scores, HTML report
├── CHANGELOG.md            # Version history and experiment findings
├── PIPELINE_DOC.md         # Detailed code documentation (Korean)
└── run_pipeline.bat        # Full pipeline runner
```

---

## Research Context

- **Domain**: Grocery retail image captioning
- **Problem**: VLM hallucination on product-specific entities
- **Approach**: Hierarchical metadata tagging (L0-L4) + lightweight VLM (3.8B edge-deployable)
- **Justification for lightweight model**: GPT-4V/Gemini Vision APIs are unsuitable for
  real-time retail operations due to latency, cost, and data privacy constraints
- **Evaluation**: Retail-CHAIR (secondary) + Multimodal Pairwise LLM Judge (primary)

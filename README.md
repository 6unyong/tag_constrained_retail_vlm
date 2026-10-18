# Bottom-Up Multimodal Architecture for Hallucination-Suppressed Retail Vision-Language Captioning

> **King's College London — MSc Data Science Dissertation (2025/26)**  
> **Author**: Junyong Park | **Supervisor**: Dr Rita Borgo  
> **Evaluation Scale**: $N = 10,686$ real-world retail store images

📄 **[한국어 논문 전체 상세 해설 보기 (THESIS_OVERVIEW.md)](./THESIS_OVERVIEW.md)**  
📊 **[Presentation Slides (PDF)](./presentation.pdf)** | 🌐 **[Interactive Presentation (HTML)](./presentation.html)**

---

## 📌 Research Overview

Vision-Language Models (VLMs) offer zero-shot product recognition for automated retail shelf monitoring without the costly retraining cycles of traditional closed-set object detectors. However, deploying lightweight edge-native VLMs (e.g., `LLaVA-Phi3 3.8B`) in chaotic, occluded store environments introduces a critical barrier: **generative object hallucination ("phantom inventory")**—where the model fabricates products in empty or ambiguous shelf spaces based on pre-trained language priors, paralysing automated replenishment systems.

Rather than imposing top-down business taxonomies (which experimentally induce *Language Prior Bias*), this dissertation proposes a fine-tuning-free **Bottom-Up Multimodal Architecture (MOP: Metadata-Grounded Object-Prompting)**. The pipeline isolates visual fact verification from business reasoning, constructing meaning progressively from unforgeable physical evidence via a **Three-Tier Safety Net**:

1. **OCR Physical Anchoring (`PaddleOCR`)**: Mechanically extracts physical text from packaging and price labels to block extrinsic brand/flavour fabrication.
2. **Zero-Shot CLIP Gatekeeper (`ViT-B/32`)**: Computes cosine similarity ($S$) to classify candidate tags into **Hard** ($S \ge 0.15$), **Soft** ($0.05 \le S < 0.15$), and **Absence** ($S < 0.05$), replacing unverified slots with **`[Ambiguous]`** (*Explicit Absence Encoding*).
3. **Deterministic Generation Restraint ($T = 0.1$ + Open+Anchor Routing)**: Dynamically routes scene-specific prompt templates via **Gower's Distance & K-Medoids ($K=2$)** while suppressing token sampling randomness.

---

## 🏆 Key Empirical Results ($N = 10,686$ Images)

Both the **Unconstrained Baseline** and the **Proposed MOP Pipeline** were evaluated using the identical local edge model (`llava-phi3`, 3.8B parameters) across 10,686 real-world retail images:

| Evaluation Dimension | Metric / Protocol | Sample Size ($N$) | Baseline (`llava-phi3`) | Proposed MOP (`llava-phi3`) | Key Finding |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Object Hallucination (RQ2)** | **Lemmatised CHAIR ($\text{L-CHAIR}_i$)** ↓ | 10,686 / 10,684 | 70.47% | **40.36%** | **-30.11%p drop (42.7% relative reduction)** |
| **Multimodal Pairwise Judge** | **Gemini 3.5 Flash Blind Judge** ↑ | 10,686 Pairs | 23.2% (2,474) | **71.1% (7,597)** | **~3:1 win ratio** (5.8% ties) |
| **Human-in-the-Loop** | **2-Rater Blind Evaluation** ↑ | 100 Pairs | 37.0% | **50.0%** | **+13.0%p lead** (13.0% ties, $\kappa = 0.036$) |
| **Archetype Clustering** | **Dunn Index / Silhouette Score** | 1,000 Samples | — | **$K = 2$ Optimal** | Dunn = **0.330**, Silhouette = **0.556** |
| **Edge-Cloud Efficiency (RQ3)** | **Local VRAM / Latency per Image** | 10,684 Images | ~5.5 GB VRAM | **~5.5 GB VRAM / 4.38s** | Deployable on consumer 6GB Edge GPU (RTX 3060) |

---

## 🏗️ 6-Stage Hybrid Pipeline Architecture

```
[Retail Shelf Imagery Dataset (N = 10,686)]
        │
        ▼
Stage 1: Image Ingestion & Hybrid Motion Blur Filter (Laplacian >= 50.0 + Adaptive FFT)
  └── src/pipeline_1_ingestion.py
        │
        ▼
Stage 2: L1 Macro Scene (CLIP Zero-Shot) & L2 Fixture Detection (Grounding DINO)
  └── src/pipeline_2_ontology.py, src/pipeline_3_dynamic_tagging.py
        │
        ▼
Stage 3: L3 Product Tagging & 3-Tier CLIP Gatekeeper (Hard >= 0.15 / Soft >= 0.05 / Absence < 0.05)
  └── src/pipeline_3b_l3_product_tagging.py (Cloud Hybrid) | src/pipeline_3b_l3_product_tagging_ollama.py (Edge)
        │
        ▼
Stage 4: L4 Operational Attributes & PaddleOCR Physical Text Anchoring
  └── src/pipeline_3c_l4_attribute_tagging.py | src/pipeline_3c_l4_attribute_tagging_ollama.py
        │
        ▼
Stage 5: MOP Routing & K-Medoids Clustering (Gower's Distance, Optimal K=2)
  └── src/pipeline_4_routing_clustering.py | src/pipeline_4_routing_clustering_ollama.py
        │
        ▼
Stage 6: Local Edge VLM Captioning — Open+Anchor Prompting (Ollama llava-phi3, Temperature = 0.1)
  └── src/pipeline_5_mop_captioning.py | src/pipeline_5_mop_captioning_ollama.py
        │
        ├──────────────────────────────────────────────────────────┐
        ▼                                                          ▼
Evaluation Suite (Dual-Verification + L-CHAIR)            Unconstrained Baseline Ablation
  ├── src/pipeline_6_eval_chair.py (L-CHAIR_i)              └── src/pipeline_ablation_baseline.py
  ├── src/pipeline_7_llm_judge.py (Pairwise LLM Judge)
  ├── src/pipeline_8_comparison_report.py (HTML Report)
  └── app.py (Flask Blind Human Annotation Web App)
```

---

## 📂 Repository Structure

```text
tag_constrained_retail_vlm/
├── README.md                                  # Project overview & quick start (English)
├── THESIS_OVERVIEW.md                         # Comprehensive chapter-by-chapter thesis guide (Korean)
├── presentation.pdf                           # Final dissertation presentation slides (PDF)
├── presentation.html                          # Reveal.js interactive presentation
├── src/                                       # Core 6-stage pipeline & evaluation scripts
│   ├── data_models.py                         # Pydantic schemas for L0-L4 hierarchical tags
│   ├── pipeline_1_ingestion.py                # Stage 1: Image ingestion & Laplacian/FFT blur filter
│   ├── pipeline_2_ontology.py                 # Stage 2: Ontology initialization
│   ├── pipeline_3_dynamic_tagging.py          # Stage 2/3A: CLIP L1 scene + Grounding DINO L2 fixtures
│   ├── pipeline_3b_l3_product_tagging.py      # Stage 3B: L3 product tagging + CLIP 3-tier gatekeeper
│   ├── pipeline_3b_l3_product_tagging_ollama.py # Stage 3B (Edge-native Ollama variant)
│   ├── pipeline_3c_l4_attribute_tagging.py    # Stage 4: PaddleOCR physical anchor extraction + CLIP
│   ├── pipeline_3c_l4_attribute_tagging_ollama.py # Stage 4 (Edge-native Ollama variant)
│   ├── pipeline_4_routing_clustering.py       # Stage 5: Gower's distance + K-Medoids (K=2) MOP routing
│   ├── pipeline_4_routing_clustering_ollama.py # Stage 5 (Edge-native Ollama variant)
│   ├── pipeline_5_mop_captioning.py           # Stage 6: Open+Anchor VLM caption generation
│   ├── pipeline_5_mop_captioning_ollama.py    # Stage 6: Local Ollama llava-phi3 captioning (T=0.1)
│   ├── pipeline_6_eval_chair.py               # Evaluation: Lemmatised CHAIR (L-CHAIR_i) computation
│   ├── pipeline_7_llm_judge.py                # Evaluation: Multimodal pairwise blind LLM-as-a-Judge
│   ├── pipeline_8_comparison_report.py        # Evaluation: Empirical report generator
│   ├── pipeline_ablation_baseline.py          # Baseline: Unconstrained VLM ablation runner
│   ├── prepare_annotation.py                  # Human-in-the-loop blind dataset preparation
│   └── utils/
│       └── async_gemini.py                    # Async Gemini API wrapper with exponential backoff
├── app.py                                     # Flask web application for blind human evaluation
├── static/ & templates/                       # Frontend UI for the human annotation tool
├── venv_setup/                                # Environment & weight setup scripts
├── requirements.txt                           # Python dependencies
└── run_pipeline.bat                           # End-to-end Windows batch pipeline runner
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
- **Local Edge VLM**: Install [Ollama](https://ollama.com/) and pull the edge model:
  ```bash
  ollama pull llava-phi3
  ```
- **Cloud API Key**: Create a `.env` file in the project root:
  ```env
  GEMINI_API_KEY=your_gemini_api_key
  ```

### 2. Running the Pipeline
```bash
# Run full pipeline interactively
run_pipeline.bat

# Or execute stages sequentially:
python src/pipeline_1_ingestion.py
python src/pipeline_3_dynamic_tagging.py
python src/pipeline_3b_l3_product_tagging.py
python src/pipeline_3c_l4_attribute_tagging.py
python src/pipeline_4_routing_clustering.py
python src/pipeline_5_mop_captioning_ollama.py

# Run Baseline & Evaluation:
python src/pipeline_ablation_baseline.py --from-mop
python src/pipeline_6_eval_chair.py
python src/pipeline_7_llm_judge.py
python src/pipeline_8_comparison_report.py
```

### 3. Running the Human Annotation Web App
```bash
python app.py
# Open http://localhost:5000 in your browser
```

---

## 📜 License & Citation

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

```bibtex
@mastersthesis{park2026bottomup,
  title   = {Bottom-Up Multimodal Architecture for Hallucination-Suppressed Retail Vision-Language Captioning},
  author  = {Park, Junyong},
  year    = {2026},
  school  = {King's College London, Department of Informatics},
  type    = {MSc Data Science Dissertation}
}
```

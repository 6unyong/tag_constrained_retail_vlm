# Antigravity Retail Captioning Pipeline — 전체 코드 문서

> **프로젝트**: KCL MSc Dissertation — Grocery Retail VLM Hallucination Control  
> **Version**: v3 (Phase 7 Complete — 10K Resilience Refactoring)  
> **최종 업데이트**: 2026-04-07

---

## 📐 전체 아키텍처 한눈에 보기

```
[HuggingFace Dataset]
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1: 데이터 수집 & 품질 필터링                  │
│  pipeline_1_ingestion.py                            │
└──────────────────────┬──────────────────────────────┘
                       │  data/processed/*.jpg
                       │  data/cache/metadata_mapped.json
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 2: 온톨로지 구축 & GS1 분류 매핑 (1회성)     │
│  pipeline_2_ontology.py → pipeline_2b_gs1_mapper.py │
└──────────────────────┬──────────────────────────────┘
                       │  data/cache/gs1_mappings.json
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3A: L1 씬 + L2 Fixture 태깅 (CLIP + DINO)   │
│  pipeline_3_dynamic_tagging.py                      │
├─────────────────────────────────────────────────────┤
│  Stage 3B: L3 제품 태깅 (Gemini API + CLIP)         │
│  pipeline_3b_l3_product_tagging.py                  │
├─────────────────────────────────────────────────────┤
│  Stage 3C: L4 속성 태깅 (PaddleOCR + CLIP)         │
│  pipeline_3c_l4_attribute_tagging.py                │
└──────────────────────┬──────────────────────────────┘
                       │  data/cache/hierarchical_tags_final.json
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4: MOP 라우팅 클러스터링 + 자동 프롬프트 생성 │
│  pipeline_4_routing_clustering.py                   │
└──────────────────────┬──────────────────────────────┘
                       │  data/cache/clustered_routes.json
                       │  data/cache/mop_prompts.json
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 5: MOP 캡셔닝 (Local Ollama VLM)             │
│  pipeline_5_mop_captioning.py                       │
└──────────────────────┬──────────────────────────────┘
                       │  data/cache/final_captions.json
                       ▼
┌─────────────────────────────────────────────────────┐
│  Stage 6: 평가                                       │
│  pipeline_6_eval_chair.py   →  Retail-CHAIR 지표     │
│  pipeline_7_llm_judge.py    →  LLM-as-a-Judge 점수  │
│  pipeline_ablation_baseline.py → 비교군(순정 LLaVA) │
└──────────────────────┬──────────────────────────────┘
                       │  data/eval_results/*.json
                       ▼
                  📊 논문 실험 결과
```

---

## 📂 파일별 코드 설명

### 기반 파일

#### `src/data_models.py`
**목적**: 파이프라인 전반에서 사용되는 Pydantic 데이터 스키마 정의.
| 클래스 | 역할 |
|---|---|
| `ImageMetadata` | 이미지 ID, 소매점, 캠페인/시즌, 블러 점수 |
| `OntologyHierarchy` | L1~L4 태깅 계층 구조 |
| `OntologyTags` | 다중 온톨로지 태그 + GS1 매핑 결합 |
| `RoutingContext` | MOP 라우팅 페르소나 및 목표 |
| `ProcessedImageRecord` | 최종 처리 레코드 통합 스키마 |

**실행 여부**: 직접 실행 불필요 (타 스크립트에서 import)

#### `src/utils/async_gemini.py`
**목적**: Gemini API 비동기 래퍼. 이미지 + 텍스트 프롬프트로 구조화된 JSON 응답을 요청.
- `generate_structured_vision_async()` — Pydantic 스키마 기반 타입 안전 응답
- `tenacity @retry` — API 에러 시 지수적 백오프(2초~30초, 최대 5회 재시도)

**실행 여부**: 직접 실행 불필요 (pipeline_3b, pipeline_4에서 import)

---

### Stage 1: 데이터 수집

#### `src/pipeline_1_ingestion.py`
**목적**: HuggingFace 스트리밍 데이터셋에서 이미지를 수집하고 블러 필터링 및 메타데이터 추출.

**실행 방법**:
```bash
python src/pipeline_1_ingestion.py           # 전체 데이터셋
python src/pipeline_1_ingestion.py --limit 100  # 일부 테스트
```

**처리 흐름**:
1. HuggingFace `kanops-open-retail-imagery` 스트리밍 연결
2. `cv2.Laplacian` 분산값으로 블러 검사 (임계값 < 50.0 → REJECTED)
3. 파일명 경로에서 소매점 이름(Asda, Tesco 등), 연도, 이벤트 추출
4. 이미지 저장 + 메타데이터 매핑 저장

**10K 안전망**: 기존 캐시 로드 → 이미 처리된 이미지 skip / 50장마다 자동 저장 / per-image try-except

| 입력 | 출력 |
|---|---|
| HuggingFace 스트림 | `data/processed/sample_N.jpg` |
| — | `data/cache/metadata_mapped.json` |

---

### Stage 2: GS1 온톨로지 구축 (1회성 실행)

#### `src/pipeline_2_ontology.py`
**목적**: 샘플 이미지에 대해 Gemini Pro로 카테고리 명사구 추출 및 원시 온톨로지 파일 생성.  
**실행 빈도**: **단 1회**

| 입력 | 출력 |
|---|---|
| `data/processed/` 샘플 이미지 | `data/cache/ontology_raw.json` |

#### `src/pipeline_2b_gs1_mapper.py`
**목적**: 원시 온톨로지를 공식 GS1 GPC (Global Product Classification) 계층에 매핑.  
**실행 빈도**: **단 1회**

| 입력 | 출력 |
|---|---|
| `data/cache/ontology_raw.json` | `data/cache/gs1_mappings.json` |

---

### Stage 3: 계층적 태깅 (L1~L4)

#### `src/pipeline_3_dynamic_tagging.py`
**목적**: L1 씬 분류(CLIP) + L2 Fixture 탐지(GroundingDINO).

| 모듈 | 모델 | 역할 |
|---|---|---|
| L1 씬 분류 | CLIP ViT-B/32 | Zero-shot 씬 레이블 분류 |
| L2 Fixture 탐지 | GroundingDINO | 텍스트 프롬프트 기반 Bounding Box 탐지 |

**태그 기준**: Hard(>0.85) / Soft(≥0.60) / Absence(<0.60)  
**실행 방법**: `python src/pipeline_3_dynamic_tagging.py` (venv_vision 필요)

| 입력 | 출력 |
|---|---|
| `data/processed/*.jpg` | `data/cache/l1_l2_tag_results.json` |

---

#### `src/pipeline_3b_l3_product_tagging.py`
**목적**: L3 제품 레이블 태깅 — Gemini API가 OCR 기반 동적 키워드를 생성하고 CLIP이 확률 점수 부여.

**처리 흐름**:
1. **PaddleOCR** — 가시 텍스트 추출 (신뢰도 > 0.5)
2. **Gemini Flash** — OCR + L1/L2 맥락 기반 10~20개 제품 키워드 생성 (용량/포장형태 제외)
3. **CLIP** — Zero-shot 확률 점수 부여 → Hard/Soft/Absence 분류

> [!IMPORTANT]
> **Gemini 이중 과금 차단**: 재실행 시 이미 L3 태그가 있는 이미지는 API 호출 자체를 skip.  
> **429 Rate Limit 방어**: tenacity 재시도 (5초~60초, 최대 5회)

**실행 방법**: `python src/pipeline_3b_l3_product_tagging.py` (venv_vision 필요)

| 입력 | 출력 |
|---|---|
| `l1_l2_tag_results.json` + `gs1_mappings.json` | `data/cache/l1_l2_l3_tag_results.json` |

---

#### `src/pipeline_3c_l4_attribute_tagging.py`
**목적**: L4 속성 태깅 (CLIP Zero-shot) + 최종 계층적 JSON 통합.

| 속성 | 임계값 |
|---|---|
| `stock_level` (완전 채움/부분 비움/완전 비움) | ≥ 0.70, 미만은 "Ambiguous" |
| `tidiness` (정돈된/어수선한) | ≥ 0.70 |
| `promotion` (프로모션 표시 있음/없음) | ≥ 0.70 |

**핵심 설계**: **Explicit Absence Encoding** — CLIP 신뢰도 < 0.70이면 `"Ambiguous"` 표시 → MOP 프롬프트가 VLM에 전달, 환각 방지

**실행 방법**: `python src/pipeline_3c_l4_attribute_tagging.py` (venv_vision 필요)

| 입력 | 출력 |
|---|---|
| `l1_l2_l3_tag_results.json` + `metadata_mapped.json` | `data/cache/hierarchical_tags_final.json` |

---

### Stage 4: MOP 라우팅

#### `src/pipeline_4_routing_clustering.py`
**목적**: 이미지를 MOP 라우팅 그룹으로 군집화 + 각 클러스터 맞춤 캡셔닝 프롬프트 자동 생성.

**Phase A — K-Means 클러스터링**:
- 5개 피처: 씬 신뢰도, fixture 수, tidiness, stock, promotion
- **Silhouette Score**로 최적 K 자동 선정 (K=2~8 전수 탐색)
  - Elbow Method 대신 사용: 단일 수치로 K 선정 → 논문 방어에 유리

**Phase B — Gemini 자동 프롬프트 생성**:
- 각 클러스터 센트로이드 특성을 Gemini에 전달
- 해당 매대 특성에 최적화된 MOP 프롬프트 자동 생성
- `{ctx_str}`, `{l3_str}` 등 FACTS 플레이스홀더 포함 → pipeline_5에서 동적 주입

**실행 방법**: `python src/pipeline_4_routing_clustering.py`

| 입력 | 출력 |
|---|---|
| `hierarchical_tags_final.json` | `data/cache/clustered_routes.json` |
| — | `data/cache/mop_prompts.json` |
| — | `data/eval_results/k_selection_report.json` |

---

### Stage 5: MOP 캡셔닝

#### `src/pipeline_5_mop_captioning.py`
**목적**: 클러스터별 MOP 프롬프트를 조립하고 로컬 Ollama VLM에 전달해 결정론적 캡션 생성.

**처리 흐름**:
1. `mop_prompts.json`에서 해당 클러스터 프롬프트 템플릿 로드
2. 이미지의 L1~L4 FACTS를 플레이스홀더에 주입
3. Ollama 로컬 VLM으로 이미지 + 프롬프트 전달 (타임아웃 30초, 온도 0.1)
4. 생성 캡션을 계층적 JSON에 병합 저장

**환각 방지 MOP 핵심 규칙**:
- `Semantic Shift Resolution` — L0 메타데이터는 배경 이해에만 사용
- `Visual Fact Priority` — 제품명은 L3+OCR 증거에서만 추출
- `Absence Handling` — Ambiguous 속성은 "불명확"으로만 표기

**지원 VLM 모델** (ollama serve 필요):
```bash
ollama pull llava               # 기본
ollama pull llama3.2-vision     # 업그레이드 (11B, 권장)
ollama pull qwen2-vl            # 대안 (리테일 OCR 특화)
```

**실행 방법**: `python src/pipeline_5_mop_captioning.py` (ollama serve 실행 중이어야 함)

| 입력 | 출력 |
|---|---|
| `clustered_routes.json` + `mop_prompts.json` | `data/cache/final_captions.json` |

---

### Stage 6: 평가

#### `src/pipeline_ablation_baseline.py`
**목적**: MOP 파이프라인과의 공정한 정량 비교를 위한 **순정 LLaVA 베이스라인 캡션** 생성.  
동일한 이미지에 아무 제약 없이 VLM이 자유롭게 캡션 생성 → 환각율 비교 기준점.

**실행 방법**: `python src/pipeline_ablation_baseline.py` (ollama serve 필요)

| 입력 | 출력 |
|---|---|
| `data/processed/*.jpg` (전체 동적 탐색) | `data/cache/baseline_captions.json` |

---

#### `src/pipeline_6_eval_chair.py`
**목적**: **Retail-CHAIR 환각 지표** 계산.

| 지표 | 설명 |
|---|---|
| `CHAIR_i` | 캡션 내 전체 명사 중 환각 명사의 비율 (%) |
| `CHAIR_s` | 전체 캡션 중 최소 1개 환각이 포함된 캡션 비율 (%) |

**매칭 알고리즘**:
- `NLTK WordNetLemmatizer` — "bottles" → "bottle" 형태소 정규화
- **Word-boundary 정규식** — `\b` 단어 경계로 "cola"가 "chocolate" 내부에 매칭되는 오탐 방지

**실행 방법**: `python src/pipeline_6_eval_chair.py`

| 입력 | 출력 |
|---|---|
| `data/cache/final_captions.json` | `data/eval_results/chair_metrics.json` |

---

#### `src/pipeline_7_llm_judge.py`
**목적**: **LLM-as-a-Judge** 다면 평가 — Gemini 3.1 Pro가 캡션의 정확성·관련성·부재 처리를 10점 척도로 평가.

| 기준 | 설명 |
|---|---|
| `accuracy_score` | Ground Truth 팩트 외 발명 없이 정확한지 (1~10) |
| `relevance_score` | 매장 관리자 입장에서 자연스러운지 (1~10) |
| `absence_handling_score` | Ambiguous 데이터를 "불명확"으로만 표기했는지 (1~10) |

> [!IMPORTANT]
> **Gemini 이중 과금 차단**: 재실행 시 이미 평가된 이미지는 API 호출 skip

**실행 방법**: `python src/pipeline_7_llm_judge.py`

| 입력 | 출력 |
|---|---|
| `data/cache/final_captions.json` | `data/eval_results/llm_judge_scores.json` |

---

## 📁 최종 결과물 구조

```
data/
├── processed/
│   └── sample_*.jpg                   # 수집된 전체 이미지 (~1만장)
│
├── cache/                             # 파이프라인 중간 결과 (체크포인트)
│   ├── metadata_mapped.json           # 이미지별 소매점·맥락 정보
│   ├── gs1_mappings.json              # GS1 GPC 온톨로지 매핑 (1회성)
│   ├── l1_l2_tag_results.json         # L1+L2 씬/Fixture 태그
│   ├── l1_l2_l3_tag_results.json      # L1+L2+L3 제품 태그
│   ├── hierarchical_tags_final.json   # L1~L4 완전 계층적 태그
│   ├── clustered_routes.json          # 라우팅 클러스터 배정 결과
│   ├── mop_prompts.json               # 클러스터별 자동 생성 MOP 프롬프트
│   ├── final_captions.json            # ✅ MOP 파이프라인 최종 캡션
│   ├── baseline_captions.json         # ✅ 순정 LLaVA 베이스라인 캡션
│   ├── error_log.txt                  # 이미지별 에러 기록
│   └── run_log.txt                    # 전체 실행 로그
│
└── eval_results/                      # 📊 논문 Table용 정량 지표
    ├── chair_metrics.json             # CHAIR_i / CHAIR_s 환각 지표
    ├── llm_judge_scores.json          # 정확성·관련성·부재처리 평균 점수
    └── k_selection_report.json        # Silhouette Score 기반 K 선정 근거
```

### 논문에 직접 사용 가능한 결과 데이터

````carousel
**`chair_metrics.json` — 환각율 비교 Table**
```json
{
  "CHAIR_i": 3.2,
  "CHAIR_s": 8.5,
  "details": [
    {"image": "sample_0.jpg", "hallucinated_nouns": [], "chair_i_score": 0.0},
    {"image": "sample_1.jpg", "hallucinated_nouns": ["asda"], "chair_i_score": 0.083}
  ]
}
```
<!-- slide -->
**`llm_judge_scores.json` — 다면 평가 Table**
```json
{
  "avg_accuracy": 8.67,
  "avg_relevance": 7.83,
  "avg_absence_handling": 9.10,
  "n_evaluated": 10000
}
```
<!-- slide -->
**`k_selection_report.json` — K 선정 근거 Figure**
```json
{
  "n_samples": 10000,
  "optimal_k": 4,
  "silhouette_scores_by_k": {
    "2": 0.3821,
    "3": 0.4910,
    "4": 0.6234,
    "5": 0.5891
  },
  "cluster_distribution": {"0": 2841, "1": 3102, "2": 2109, "3": 1948}
}
```
````

---

## 🚀 실행 방법

### 원클릭 전체 실행 (권장)

```powershell
# Step 1: 별도 터미널에서 Ollama 실행 (캡셔닝에 필요)
ollama serve

# Step 2: 루트 폴더에서 배치 파일 실행
.\run_pipeline.bat
```

### 개별 스테이지 실행 (venv_vision 내에서)

```powershell
.\venv_setup\venv_vision\Scripts\activate

python src/pipeline_1_ingestion.py              # Stage 1: 이미지 수집 (전체)
python src/pipeline_3_dynamic_tagging.py        # Stage 2: L1+L2 태깅
python src/pipeline_3b_l3_product_tagging.py    # Stage 3: L3 태깅 (Gemini API)
python src/pipeline_3c_l4_attribute_tagging.py  # Stage 4: L4 태깅
python src/pipeline_4_routing_clustering.py     # Stage 5: 라우팅 + 프롬프트 생성
python src/pipeline_5_mop_captioning.py         # Stage 6: MOP 캡셔닝
python src/pipeline_ablation_baseline.py        # Stage 7: 베이스라인 (비교군)
python src/pipeline_6_eval_chair.py             # Stage 8: CHAIR 평가
python src/pipeline_7_llm_judge.py              # Stage 9: LLM Judge 평가
```

> [!TIP]
> **중단 후 재개**: 동일 명령어를 다시 실행하면 됩니다. 모든 스크립트가 이미 처리된 이미지를 자동으로 건너뜁니다.

> [!WARNING]
> `pipeline_3`, `3b`, `3c`는 반드시 `venv_vision` 환경에서 실행해야 합니다.  
> Gemini API가 호출되는 스테이지 (`3b`, `4`, `7`)는 재실행 시 완료된 항목에 대해 API를 재호출하지 않습니다.

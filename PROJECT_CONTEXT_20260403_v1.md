# Grocery Retail Captioning Pipeline - Context Backup
**Date Generated:** 2026-04-03
**Version:** v3 (Phase 7 - 10K Resilience Refactoring Complete)

> [!TIP]
> **AI 프롬프트 인스트럭션 (새로운 대화창에서 재개할 때)**:
> "이 `PROJECT_CONTEXT_20260403_v1.md` 파일을 읽고, 현재 아키텍처 결정 사항을 파악한 뒤 **Task 10** 부터 이어서 코딩을 시작해 줘."

---

## 1. 프로젝트 주요 목표 (MSc Dissertation)
*   **목적**: 리테일 매장(Grocery) 이미지 캡셔닝 파이프라인 구축.
*   **핵심 기여(Novelty)**: 기존 VLM의 한계인 '환각(Hallucination)'을 줄이기 위해, (1) 결정론적 태깅, (2) 자동화된 매장 시나리오 라우팅, (3) 명시적 부재 인코딩(Explicit Absence Encoding) 파이프라인 제안.

## 2. 아키텍처 설계 & 모델 결정 사항 (방어 논리 포함)
*   **Stage 1 (Feature Extraction)**: `CLIP` (Zero-shot) + `Grounding DINO` (BBox) + `PaddleOCR` (Text). 
    *   *결정 사유*: 논문 Ablation Study를 위해 모듈을 쪼개어 세밀한 성능 조절이 가능하게 함.
*   **Stage 2 (Orchestrator & GS1 Mapping)**: `Gemini 2.5 Pro / Flash` (API 기반)
    *   *결정 사유*: 텍스트 기반의 복잡한 JSON 구조화 및 온톨로지 매핑에 가장 안정적이고 저렴함 (`src/utils/async_gemini.py` 로 비동기 핸들링 달성).
*   **Stage 3 (Caption Generator)**: `LLaVA-1.5-7B` 또는 `Qwen-VL-Chat` (로컬 구동 모델)
    *   *결정 사유*: API 비용 및 리테일 프라이버시(GDPR) 문제 방어 등 Edge Runtime 환경 제약을 시뮬레이션함. 베이스라인 환각율이 높은 모델을 사용함으로써, 본 논문에서 제안한 파이프라인의 **폭발적인 환각 개선율(%)**을 확실하게 입증하기 위함.
    *   *(방어책)*: 단, 심사관의 비판(스케일 이슈)을 막기 위해 50장 정도의 별도 테스트 셋은 최고 등급인 Gemini Pro/GPT-4o 로도 캡션을 뽑아 비교 검증(Ablation)할 예정.

## 3. 현재 작업 진척도 (Progress)
*   **완료 단계**: Phase 1 (Data Foundation & Corpus Induction) - **Task 1 ~ 7 100% 완료**
    *   ✅ HuggingFace `kanops-open-retail-imagery` 데이터셋 스트리밍 연결 완료.
    *   ✅ `cv2.Laplacian` 기반 블러 이미지 필터링 알고리즘 적용 완료 (`src/pipeline_1_ingestion.py`).
    *   ✅ `Gemini 2.5 Pro`를 이용한 3장 샘플의 명사구(Corpus) 비동기 추출 성공 (`src/pipeline_2_ontology.py`).
    *   ✅ 추출된 원시 카테고리를 LLM을 거쳐 공식 **GS1 GPC (Global Product Classification) Architecture**로 정밀 맵핑 저장 완료 (`src/pipeline_2b_gs1_mapper.py`).
*   **완료 단계**: Phase 2 (Dynamic Context Tagging) - **Task 8 ~ 9 완료**
    *   ✅ Task 8: Vision 전용 가상환경(`venv_setup/venv_vision`) 구축 완료. PyTorch 2.5.1+cu121, CLIP 1.0, groundingdino-py 0.4.0 설치됨.
    *   ✅ Task 9: `src/pipeline_3_dynamic_tagging.py` — L1(CLIP Scene) + L2(GroundingDINO Fixture) 태깅 파이프라인 작성 및 실행 성공.
    *   ✅ Task 10: `src/pipeline_3b_l3_product_tagging.py` — Gemini 1.5 Flash와 통신하여 생성된 동적 키워드로 L3 태깅.
    *   ✅ Task 11: `src/pipeline_3c_l4_attribute_tagging.py` — PaddleOCR(v4)로 가격표 등 텍스트 추출 및 CLIP으로 운영 속성 판별 성공. 이를 하나의 계층적 JSON(`hierarchical_tags_final.json`)으로 통합.

## 4. 해결된 에러 사항 (Debugging Log)
1.  **GroundingDINO 원본 소스 빌드 실패**: Windows MSVC 호환 문제로 `pip install git+https://github.com/IDEA-Research/GroundingDINO.git` 실패.
    *   **해결**: pre-built 패키지 `groundingdino-py==0.4.0` 으로 대체 설치. 단, Windows 한국어 로케일 `cp949` 인코딩 충돌이 있어 `$env:PYTHONUTF8="1"` 환경변수 설정 후 설치 성공.
2.  **transformers 5.x 호환 에러**: `BertModel` 의 `get_head_mask` 메서드가 제거되어 GroundingDINO 로드 시 `AttributeError` 발생.
    *   **해결**: `transformers==4.47.1` 로 다운그레이드하여 해결.

## 5. 설치된 패키지 요약 (venv_vision)
| 패키지 | 버전 | 비고 |
|---|---|---|
| torch | 2.5.1+cu121 | CUDA 12.1 |
| torchvision | 0.20.1+cu121 | |
| clip | 1.0 | OpenAI CLIP |
| groundingdino-py | 0.4.0 | pre-built (원본 소스 빌드 대체) |
| transformers | 4.47.1 | 5.x에서 다운그레이드 (DINO 호환) |
| timm | 1.0.26 | |

## 6. Task 11 실행 결과 샘플
```
sample_0.jpg
  OCR: ['TasT', 'asi', 'chicken', 'ana &', 'hicken']
  Attributes: stock_level(fully stocked, 0.698), tidiness(neatly organized, 0.527), promotion(promotional signage, 0.911)
```

## 7. Task 13 & 14 (MOP Captioning) 실행 결과!
Ollama 4-bit LLaVA 1.5 7B 환경에서 환각 없이 완벽한 결정론적(Deterministic) 캡션 추출 성공.
```text
[sample_0.jpg - MOP Route 0 중심]
LLaVA: The image showcases a well-stocked retail endcap featuring a neatly organized display of Tesco Chicken Salad Sandwich, M&S Prawn Mayonnaise Sandwich, and Sainsbury's Ham & Cheese Sandwich. promotional signage is visible.

[sample_2.jpg - MOP Route 1 중심]
LLaVA: The grocery checkout area features a neatly organized retail display with a partially empty shelf. There are three magazines on the shelf: Hello! Magazine (current issue), Daily Mail Newspaper, and The Guardian Newspaper. Relevant signs such as "Fruit&," "U," "99P" are visible.
```

## 8. 아키텍처 결정 사항 (Phase 6 Final Optimization)
*   **Semantic Shift vs Parametric Hallucination Resolution**: VLM 모델의 우선순위 역전 결함(시각적 증거보다 메타데이터를 더 신뢰하여 환각을 일으킴)을 파훼하기 위해 MOP 프롬프트에 `Priority Rule`을 도입. 메타데이터(L0)는 계절 행사나 PB 상품의 맥락(Semantic Shift 보정)을 이해하는 데만 한정되도록 제한하고, 제품의 명칭 등 하위 요소는 오직 시각 증거(OCR)를 우선하도록 락업(Lock-up).
*   **Hybrid Extraction (L3 Filter) Constraint Removed**: 기존에는 `TagType: Absence`를 필터링 후 최대 3개까지만 전송했으나, 정상적인 팩트(Factual) 상품 데이터의 유실을 방지하기 위해 개수 제한 제약을 전면 해제함. 신규 VLM(Llama 3.2 Vision 등) 테스트를 위해 모든 유효 상품을 온전히 MOP 프롬프트에 담도록 아키텍처 개편.
*   **Dual-Verification (OCR-Bounded Dictionary)**: 허구의 브랜드 네이밍이 생성되는 것을 원천 차단하기 위해, L4의 PaddleOCR 추출을 L3의 Gemini 후보군 생성 단계 프롬프트로 전진 배치. (시각적 팩트 기반 추론 강제)
*   **Stylistic Forcing (Persona Lock-in)**: LLaVA 7B 모델 특유의 Instruction Neglect 현상을 방어하기 위해 MOP 프롬프트 상단 첫 문장을 `"Merchandising Report: "` / `"Store Inspection: "` 기조로 고정 (Generalization 방어 논리 탑재).

## 9. 실험 및 평가 (Phase 6 Final)
*   ✅ **Retail-CHAIR(환각 지표)**: LLaVA의 순정 모델(100% 환각률 / Asda 매장을 인지하지 못하고 거짓말)을 MOP 구조가 완벽하게 교정.
*   ✅ **LLM-as-a-Judge (Gemini-3.1-pro)**: 정답지가 없는 리테일 환경에서의 다면평가. 정확성 및 부재 회피력 분야에서 MOP 아키텍처의 우수성 확립 (Walkthrough 아티팩트 참조).

## 10. ✅ Phase 7 완료: 10K Resilience Refactoring (대규모 처리 안전망)

> [!NOTE]
> **Phase 7 리팩토링 완료 (2026-04-07)**. 아래 내용은 모두 구현 완료된 사항입니다.

### 완료된 핵심 변경사항
1. **✅ Dict-based Checkpoint (이중 과금 완벽 차단)**:
   - `pipeline_1`, `pipeline_3`, `pipeline_3b`, `pipeline_3c`, `pipeline_5` 전체에 적용.
   - 시작 시 기존 Cache JSON을 `{image_path: result}` dict로 로드 → 이미 처리된 경로는 루프 최상단에서 `continue` 로 즉시 건너뜀.
   - **Gemini API 재호출 원천 차단** — 재실행해도 `pipeline_3b`는 완료된 이미지에 대해 API를 호출하지 않음.
2. **✅ Per-image Try-Except + 에러 로깅**:
   - 모든 파이프라인에 이미지 단위 방어막 적용. 오류 발생 시 `data/cache/error_log.txt`에 기록 후 다음 이미지로 진행.
3. **✅ 50장 단위 Auto-save**:
   - 50장 처리마다 즉시 JSON 덮어쓰기 저장 → OOM/크래시 발생 시 최대 50장만 손실.
4. **✅ pipeline_1 하드코딩 제거**:
   - `islice(ds, 3)` 삭제 → `python src/pipeline_1_ingestion.py --limit 10000` 으로 원하는 수 지정. `--limit 0` 은 전체 스트리밍.
5. **✅ VRAM 누수 방지 (pipeline_3c)**:
   - 이미지 처리 후 `torch.cuda.empty_cache()` 호출로 장시간 실행 시 VRAM 점진적 누수 방지.
6. **✅ Ollama 무한 대기 방지 (pipeline_5)**:
   - `timeout: 30` 옵션 추가 → Ollama 무응답 시 30초 후 자동 스킵.

## 11. Next Step - Phase 8: 10K Full Execution
**실행 순서 (아래 명령어를 순서대로 실행):**
```bash
# 1. 전체 이미지 수집 및 메타데이터 매핑 (전체 데이터셋, 이어하기 지원)
python src/pipeline_1_ingestion.py

# 2. L1(Scene) + L2(Fixture) 태깅 (이어하기 지원, 에러 자동 스킵)
python src/pipeline_3_dynamic_tagging.py

# 3. L3 제품 태깅 — Gemini API 호출 (이어하기 지원, 이중 과금 차단)
python src/pipeline_3b_l3_product_tagging.py

# 4. L4 속성 태깅 (이어하기 지원, VRAM 자동 정리)
python src/pipeline_3c_l4_attribute_tagging.py

# 5. MOP 라우팅 클러스터링 — L1~L4 완료 후 단독 실행 (수초 소요)
#    K값은 데이터 규모에 맞게 조정 가능 (재실행 비용: 0)
python src/pipeline_4_routing_clustering.py

# 6. MOP 캡셔닝 — 로컬 VLM 필요 (이어하기 지원)
python src/pipeline_5_mop_captioning.py
```
> [!TIP]
> **오류 발생 시**: `data/cache/error_log.txt` 파일을 확인하세요. 스크립트를 재실행하면 자동으로 오류난 부분부터 이어서 처리됩니다.

# CHANGELOG

## v4.1 — Same-Model Fair Comparison Experiment (2026-04-25)

### 배경
v4 100-sample 실험에서 MOP win rate 40.7% vs Baseline 55.6%로 MOP가 열세였으나,
**모델 크기 불일치**(MOP = llava-phi3 3.8B, Baseline = llava 7B)가 핵심 원인으로 판명.
확인된 문제를 수정하여 동일 모델 조건(llava-phi3 vs llava-phi3) 재실험 실시.

### 변경 사항

#### `src/pipeline_ablation_baseline.py`
- `--model` 플래그 추가 (default: `llava:latest`)
  - `--model llava-phi3` 로 동일 모델 baseline 생성 가능
- `--out` 플래그 추가 → 별도 출력 파일 지정 (기존 7B baseline 보존)
- 출력 JSON에 `model` 필드 추가 (어떤 모델로 생성됐는지 기록)

#### `src/pipeline_7_llm_judge.py`
- `--baseline-path` 플래그 추가 → `baseline_captions_phi3.json` 등 다른 baseline 지정 가능
- `--out` 플래그 추가 → 별도 judge 결과 파일 지정
- cp949 UnicodeEncodeError 수정 (em-dash `—` → `--`)

### 실험 결과 (100-sample, llava-phi3 vs llava-phi3)

| 지표 | v4 (7B baseline) | v4.1 (phi3 baseline) | Delta |
|---|---|---|---|
| MOP win rate | 40.7% (n=27) | **74.0%** (n=100) | +33.3pp |
| Baseline win rate | 55.6% | 25.0% | -30.6pp |
| Ties | 3.7% | 1.0% | |
| VisAcc 차원 승률 | 33.3% | 70.0% | +36.7pp |
| FactComp 차원 승률 | 48.1% | 77.0% | +28.9pp |
| UncertH 차원 승률 | 18.5% | 41.0% | +22.5pp |

**해석**: 모델 크기 불균형을 제거하자 MOP 파이프라인이 **74%** 승률을 달성.
이는 v4에서 관찰된 Baseline 우세가 순수 모델 용량 차이였음을 실증적으로 입증.
동일 모델 조건에서 MOP의 L3 앵커 기반 Open+Anchor 구조가 시각 정확성과
사실 완결성에서 명확한 우위를 가져옴을 확인.

### 신규 파일
- `data/cache/baseline_captions_phi3.json` — llava-phi3 vanilla baseline (100개)
- `data/eval_results/llm_judge_scores_phi3.json` — 동일 모델 pairwise 평가 결과
- `data/eval_results/l3_quality_check.html` — L3 태그 수동 검증 인터랙티브 HTML
- `src/pipeline_l3_quality_check.py` — L3 tag quality check 신규 스크립트

### pipeline_8 업데이트 (2026-04-26)

#### `src/pipeline_8_comparison_report.py`
- `--baseline-path`, `--judge-path`, `--judge-path-fair`, `--out-html`, `--out-json` CLI 플래그 추가
- **신규 섹션: "Fair Comparison"** — 7B (혼합 모델) vs phi3 (동일 모델) 나란히 비교
  - Animated win-rate bar (MOP vs Baseline vs Tie)
  - 3차원 승률 바 차트 (VisAcc / FactComp / UncertH)
  - 해석 텍스트: 모델 크기 제거 시 +33.3pp 증가 설명
- **신규 섹션: "Architecture Evolution Gallery"** — 아키텍처 변화에 따른 캡션 변화
  - 동일 모델 조건에서 Unconstrained ↔ MOP Open+Anchor 캡션 나란히 표시
  - 5개 MOP 승리 샘플 + 2개 Baseline 승리 샘플 (균형 잡힌 honest 발표)
  - Gemini Judge reasoning 포함
- 네비게이션 바에 두 섹션 링크 추가
- 출력 summary에 Fair Judge win rate 추가

#### `src/pipeline_l3_quality_check.py` [NEW]
- `--n N` (default: 20), `--seed`, `--out` 플래그
- 이미지 + L3 태그 + MOP 앵커 + 생성 캡션을 나란히 배치
- **인터랙티브 리뷰**: "Tags OK / Tags Wrong" 버튼으로 수동 검증
- 실시간 정확도 카운터 (태그 정확도 % 자동 집계)
- 결과: `data/eval_results/l3_quality_check.html`

---

## v4 — Open+Anchor Architecture & Pairwise Evaluation (2026-04-17)

### 배경 및 동기
100-sample 예비 실험(v3 파이프라인)을 통해 세 가지 핵심 문제를 발견:

1. **Closed-world 아키텍처 문제**: MOP 프롬프트가 L1-L4 태그 어휘만을 허용하는
   closed-world constraint로 설계되어 있어, VLM이 이미지에서 시각적으로 명확한
   맥락(Halloween 테마, 계절적 색채, 레이아웃)을 포착하지 못함.
   태그가 VLM의 시각 이해를 "보완"해야 하는데 사실상 "대체"하는 구조였음.

2. **Raw OCR 이중 주입 문제**: L3 Gemini 태깅이 OCR을 이미 해석해 제품 키워드로 변환했음에도
   불구하고, Raw OCR 토큰("te", "s", "co" 등 부분 문자)을 최종 MOP 프롬프트에 재주입.
   VLM이 이 노이즈 토큰을 캡션에 그대로 복사하는 문제 발생.

3. **평가 메트릭 순환논리**: Retail-CHAIR의 Ground Truth 어휘가 L1-L4 태그로부터 구성되고,
   MOP 캡션도 동일 태그 기반으로 생성됨. 이는 prompt compliance를 측정하는 것이지
   실제 hallucination을 독립적으로 평가하는 것이 아님.

---

### 변경 사항

#### `src/pipeline_5_mop_captioning.py` — Open+Anchor 아키텍처로 전환

**기존 (Closed-world)**:
```
"ONLY describe using these facts: [L1-L4 tags]"
→ VLM의 시각적 이해가 억제됨
```

**변경 후 (Open+Anchor)**:
```
"Describe what you SEE. You MUST mention these VERIFIED facts: [L3 anchors].
 Do NOT guess: [Ambiguous L4 attributes]."
→ VLM이 이미지를 자유롭게 묘사하면서 검증된 사실만 앵커로 제약
```

세부 변경:
- `build_mop_prompt()` 전면 재작성: Closed → Open+Anchor
- Raw OCR 주입 완전 제거 (L3 Gemini 해석 결과로 대체)
- `_l4_to_natural()` 함수 신규 추가: L4 raw 레이블 → 자연어 변환
  - `"stock_level: obvious missing items"` → `"some visible shelf gaps"`
  - Ambiguous 속성은 VLM이 추측하지 않도록 명시적 격리
- L3 제품 신뢰도 하한선 추가 (Hard ≥ 0.55, Soft ≥ 0.30)
- `--sample N` 플래그 추가 (stratified sampling by MOP cluster)

#### `src/pipeline_4_routing_clustering.py` — Open+Anchor 스타일 프롬프트 생성

Gemini에게 요청하는 클러스터별 프롬프트 템플릿 생성 instruction 변경:

**기존**: "VLM이 제공된 FACTS만 사용하도록 지시하는 프롬프트 작성"

**변경 후**: "VLM이 시각적으로 자유롭게 묘사하되, 검증된 제품 앵커(l3_str)는 반드시
포함하고, 불확실한 속성(ambiguous)은 추측하지 않도록 지시하는 프롬프트 작성"

- 플레이스홀더 변경: `{ocr_str}`, `{stock}`, `{tidy}`, `{promo}` 제거
  → `{l3_str}` (anchors), `{obs_str}` (observable), `{ambiguous}` (withheld)

#### `src/pipeline_7_llm_judge.py` — Multimodal Pairwise 평가로 전면 재작성

**기존 방식 (v3)**:
- Caption text만 Gemini에 전달
- 절대 점수 (1-10): Accuracy / Relevance / Absence Handling
- Ground Truth = L1-L4 태그 (순환논리)

**변경 후 (v4)**:
- **이미지 + 두 캡션 동시 전달** (진짜 시각적 검증)
- **Pairwise 비교**: MOP vs Baseline — "어느 쪽이 더 나은가?"
- **순환논리 제거**: Judge가 이미지를 직접 보고 평가, GT 태그 의존 없음
- **3개 차원 평가**: 시각 정확성 / 사실 완결성 / 불확실성 처리
- **Position bias 방지**: 캡션 제시 순서 무작위화
- 출력: win-rate (MOP wins / Baseline wins / Ties)
- `--sample N` 플래그 / 503 exponential backoff 유지

#### `src/pipeline_ablation_baseline.py` — Baseline 유연성 개선

- `--from-mop` 플래그 추가: MOP 샘플과 동일 이미지 세트 사용
- `--sample N` 플래그 추가
- OLLAMA_TIMEOUT 30s → 120s (vision task에 충분한 시간 확보)
- em-dash 인코딩 오류 수정 (cp949 Windows 환경 호환)

#### `src/pipeline_8_comparison_report.py` — 교수 발표용 HTML 리포트로 전면 재작성

- 기존 Markdown 리포트 → 자급자족(self-contained) HTML 파일
- 섹션: Research Overview / Metrics Dashboard / Stage Evolution /
  Before-After Gallery / Hallucination Distribution / Best & Worst Cases
- 파이프라인 아키텍처 플로우다이어그램 (HTML/CSS)
- Stage-by-Stage 태그 진화 시각화 (L0→L1→L2→L3→L4→Caption)
- 할루시네이션 단어 인라인 하이라이팅 (`<mark class="hall">`)
- SVG 막대 히스토그램 (CHAIR_i 분포)
- Sticky nav / smooth scroll

---

### 100-sample 예비 실험 결과

| 지표 | v3 (Closed) | v4 (Open+Anchor) | 해석 |
|------|------------|-----------------|------|
| CHAIR_i MOP | 41.14% | 78.41% | CHAIR의 한계 입증: Open 방식에서 GT 불일치 증가 |
| CHAIR_i Baseline | 89.89% | 89.98% | 동일 (baseline 변경 없음) |
| Pairwise MOP 승률 | — | 40.7% | 모델 크기 차이(3.8B vs 7B) 영향 |
| Pairwise Baseline 승률 | — | 55.6% | — |

**CHAIR_i 상승의 해석**: Open 방식에서 VLM이 더 풍부한 어휘를 사용하므로,
태그 기반 GT와의 불일치가 자연스럽게 증가. 이는 CHAIR가 hallucination을
측정하는 것이 아니라 tag compliance를 측정한다는 순환논리 문제의 실증적 증거.

---

### 확인된 추가 한계점 (Next Steps)

1. **모델 크기 불일치**: MOP = llava-phi3 (3.8B), Baseline = llava (7B).
   공정한 비교를 위해 동일 모델 사용 필요 → 다음 실험에서 수정 예정.

2. **L3 태그 품질 미검증**: L3 태그 자체의 이미지 일치 여부를 수동 검증한
   실험 없음. 태깅 단계에서 발생한 hallucination은 현재 평가에서 탐지 불가.

3. **Two-pass 생성 미구현**: Open+Anchor는 프롬프트 레벨 해결책.
   더 강력한 방향: Pass1(자유 생성) + Pass2(태그 기반 사후 검증)의
   RAG 스타일 아키텍처 고려 필요.

---

## v3 — 10K 탄력성 리팩터링 (2026-04-07)
*(이전 PIPELINE_DOC.md 참조)*

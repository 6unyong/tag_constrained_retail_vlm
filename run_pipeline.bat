@echo off
REM ============================================================
REM  Antigravity Pipeline - Full Execution Runner
REM  Runs all stages in sequence using the venv_vision environment.
REM  Supports resume: already-processed images are automatically skipped.
REM  Errors are logged to data/cache/error_log.txt
REM ============================================================

setlocal enabledelayedexpansion

REM -- Force UTF-8 encoding for all Python output (prevents UnicodeEncodeError on Korean Windows)
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM -- Configuration ------------------------------------------------------------
set VENV=venv_setup\venv_vision\Scripts\activate.bat
set SRC=src
set LOG=data\cache\run_log.txt

REM -- Pre-flight checks --------------------------------------------------------
if not exist %VENV% (
    echo [ERROR] venv_vision not found at %VENV%
    echo         Run venv_setup\setup_vision_env.bat first.
    pause
    exit /b 1
)

if not exist data\cache mkdir data\cache
if not exist data\eval_results mkdir data\eval_results

echo [%date% %time%] === PIPELINE FULL RUN STARTED === >> %LOG%
echo ============================================================
echo  Antigravity Retail Captioning Pipeline - Full Execution
echo  Log: %LOG%
echo  Resume supported: already-processed images will be skipped.
echo ============================================================
echo.

REM -- Activate venv_vision -----------------------------------------------------
call %VENV%
echo [OK] venv_vision activated.

REM -- Auto-install all required packages ---------------------------------------
echo Checking and installing required packages (with version constraints)...
pip install -r requirements.txt -c constraints.txt --quiet >> %LOG% 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Package installation failed. A version conflict was detected.
    echo         Check %LOG% for details.
    echo         Common fix: transformers was upgraded beyond 4.47.1
    echo         Run: pip install transformers==4.47.1 --force-reinstall
    pause
    exit /b 1
) else (
    echo [OK] All packages installed within safe version constraints.
)

REM -- Verify critical version locks --------------------------------------------
python %SRC%\check_versions.py >> %LOG% 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Version conflict detected. Check %LOG% for details.
    echo         Common fix: pip install transformers==4.47.1 --force-reinstall
    pause
    exit /b 1
) else (
    echo [OK] Version checks passed.
)
echo.


REM -- Stage 1: Image Ingestion -------------------------------------------------
echo [STAGE 1/7] Image Ingestion (full dataset, resumable)...
echo [%date% %time%] STAGE 1 started >> %LOG%
python %SRC%\pipeline_1_ingestion.py >> %LOG% 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Stage 1 exited with errors. Check %LOG%. Continuing...
) else (
    echo [OK] Stage 1 complete.
)
echo.

REM -- Stage 2: L1 + L2 Tagging (CLIP + GroundingDINO) -------------------------
echo [STAGE 2/7] L1 Scene + L2 Fixture Tagging (resumable)...
echo [%date% %time%] STAGE 2 started >> %LOG%
python %SRC%\pipeline_3_dynamic_tagging.py >> %LOG% 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Stage 2 exited with errors. Check %LOG%. Continuing...
) else (
    echo [OK] Stage 2 complete.
)
echo.

REM -- Stage 3: L3 Product Tagging (Gemini API + CLIP) -------------------------
echo [STAGE 3/7] L3 Product Tagging - Gemini API calls (resumable, double-billing guarded)...
echo [%date% %time%] STAGE 3 started >> %LOG%
python %SRC%\pipeline_3b_l3_product_tagging.py >> %LOG% 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Stage 3 exited with errors. Check %LOG%. Continuing...
) else (
    echo [OK] Stage 3 complete.
)
echo.

REM -- Stage 4: L4 Attribute Tagging (PaddleOCR + CLIP) ------------------------
echo [STAGE 4/7] L4 Attribute Tagging (resumable)...
echo [%date% %time%] STAGE 4 started >> %LOG%
python %SRC%\pipeline_3c_l4_attribute_tagging.py >> %LOG% 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Stage 4 exited with errors. Check %LOG%. Continuing...
) else (
    echo [OK] Stage 4 complete.
)
echo.

REM -- Stage 5: MOP Routing + Auto Prompt Generation ----------------------------
echo [STAGE 5/7] MOP Routing Clustering + Gemini Prompt Generation...
echo [%date% %time%] STAGE 5 started >> %LOG%
python %SRC%\pipeline_4_routing_clustering.py >> %LOG% 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Stage 5 failed. Routing is required for captioning. Check %LOG%.
    echo         Fix the error and re-run this script (previous stages will be skipped).
    pause
    exit /b 1
) else (
    echo [OK] Stage 5 complete. Cluster prompts generated.
)
echo.

REM -- Stage 6: MOP Captioning (Ollama local VLM) ------------------------------
echo [STAGE 6/7] MOP Captioning via local Ollama VLM (resumable)...
echo            Make sure 'ollama serve' is running in another terminal!
echo [%date% %time%] STAGE 6 started >> %LOG%
python %SRC%\pipeline_5_mop_captioning.py >> %LOG% 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Stage 6 exited with errors. Check %LOG%. Continuing to evaluation...
) else (
    echo [OK] Stage 6 complete.
)
echo.

REM -- Stage 7: Evaluation (CHAIR + LLM Judge) ---------------------------------
echo [STAGE 7/7] Running Retail-CHAIR evaluation...
echo [%date% %time%] STAGE 7a (CHAIR) started >> %LOG%
python %SRC%\pipeline_6_eval_chair.py >> %LOG% 2>&1
if %errorlevel% neq 0 (
    echo [WARN] CHAIR evaluation failed. Check %LOG%.
) else (
    echo [OK] CHAIR evaluation complete.
)

echo.
echo Running LLM-as-a-Judge evaluation (Gemini API, resumable)...
echo [%date% %time%] STAGE 7b (LLM Judge) started >> %LOG%
python %SRC%\pipeline_7_llm_judge.py >> %LOG% 2>&1
if %errorlevel% neq 0 (
    echo [WARN] LLM Judge evaluation failed. Check %LOG%.
) else (
    echo [OK] LLM Judge evaluation complete.
)

REM -- Stage 8: MOP vs Baseline Comparison Report ------------------------------
echo.
echo [STAGE 8/8] Generating MOP vs Baseline Comparison Report...
echo [%date% %time%] STAGE 8 started >> %LOG%
python %SRC%\pipeline_8_comparison_report.py >> %LOG% 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Comparison report failed. Check %LOG%.
) else (
    echo [OK] Comparison report generated.
)

echo.
echo ============================================================
echo  ALL STAGES COMPLETE
echo  Results:
echo    - Final captions:    data\cache\final_captions.json
echo    - CHAIR metrics:     data\eval_results\chair_metrics.json
echo    - LLM Judge scores:  data\eval_results\llm_judge_scores.json
echo    - K-selection:       data\eval_results\k_selection_report.json
echo    - Error log:         data\cache\error_log.txt
echo    - Full run log:      %LOG%
echo ============================================================
echo [%date% %time%] === PIPELINE FULL RUN COMPLETE === >> %LOG%

pause

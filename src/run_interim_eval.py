"""
run_interim_eval.py
────────────────────
Runs the full evaluation stack (CHAIR → LLM-Judge → Comparison Report)
against the interim captions, without touching the real final_captions.json.

Steps:
  1. Temporarily symlinks / copies final_captions_interim.json → final_captions.json
  2. Runs pipeline_6 (Retail-CHAIR)
  3. Runs pipeline_7 (LLM-Judge)          -- requires GOOGLE_API_KEY
  4. Runs pipeline_8 (Comparison Report)
  5. Restores original final_captions.json if it existed

Usage:
    python src/run_interim_eval.py [--skip-judge] [--skip-report]

Flags:
    --skip-judge    Skip pipeline_7 (saves Gemini API quota)
    --skip-report   Skip pipeline_8
"""
import os, sys, shutil, argparse, subprocess

CACHE_DIR       = "data/cache"
INTERIM_PATH    = os.path.join(CACHE_DIR, "final_captions_interim.json")
LIVE_PATH       = os.path.join(CACHE_DIR, "final_captions.json")
LIVE_BACKUP     = os.path.join(CACHE_DIR, "final_captions.json.bak")

PYTHON = sys.executable


def run_stage(name: str, script: str):
    print(f"\n{'='*60}")
    print(f"  Running {name}")
    print(f"{'='*60}")
    result = subprocess.run([PYTHON, script], capture_output=False)
    if result.returncode != 0:
        print(f"[WARN] {name} exited with code {result.returncode}. Continuing…")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-judge",  action="store_true", help="Skip LLM-Judge (saves Gemini API quota)")
    parser.add_argument("--skip-report", action="store_true", help="Skip comparison report")
    args = parser.parse_args()

    # ── Validate interim file ──────────────────────────────────────────────
    if not os.path.exists(INTERIM_PATH):
        print(f"[ERROR] {INTERIM_PATH} not found.")
        print("        Run:  python src/pipeline_5_interim_captions.py  first.")
        sys.exit(1)

    import json
    with open(INTERIM_PATH, encoding="utf-8") as f:
        interim_data = json.load(f)
    print(f"[INFO] Interim captions loaded: {len(interim_data)} images")

    # ── Swap files ─────────────────────────────────────────────────────────
    backed_up = False
    if os.path.exists(LIVE_PATH):
        shutil.copy2(LIVE_PATH, LIVE_BACKUP)
        backed_up = True
        print(f"[INFO] Backed up original final_captions.json → {LIVE_BACKUP}")

    shutil.copy2(INTERIM_PATH, LIVE_PATH)
    print(f"[INFO] Interim captions → {LIVE_PATH} (evaluation target)")

    try:
        # ── Stage 6: Retail-CHAIR ─────────────────────────────────────────
        run_stage("Pipeline 6 - Retail-CHAIR", "src/pipeline_6_eval_chair.py")

        # Stage 7: LLM-Judge (optional, costs Gemini quota)
        if not args.skip_judge:
            run_stage("Pipeline 7 - LLM-Judge", "src/pipeline_7_llm_judge.py")
        else:
            print("\n[SKIP] Pipeline 7 (LLM-Judge) - use --skip-judge=False to enable")

        # Stage 8: Comparison Report
        if not args.skip_report:
            run_stage("Pipeline 8 - Comparison Report", "src/pipeline_8_comparison_report.py")
        else:
            print("\n[SKIP] Pipeline 8 (Comparison Report)")

    finally:
        # ── Always restore original ────────────────────────────────────────
        if backed_up:
            shutil.copy2(LIVE_BACKUP, LIVE_PATH)
            print(f"\n[INFO] Restored original final_captions.json from backup.")
        else:
            os.remove(LIVE_PATH)
            print(f"\n[INFO] Removed temporary final_captions.json (was empty before).")

    print("\n[DONE] Interim evaluation complete.")
    print("       Results in: data/eval_results/")


if __name__ == "__main__":
    main()

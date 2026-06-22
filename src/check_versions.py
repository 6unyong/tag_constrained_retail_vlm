"""
Version sanity checks for the pipeline.
Called by run_pipeline.bat before any stages run.
Exits with code 1 if a critical version conflict is detected.
"""
import sys

def check_transformers():
    try:
        import transformers
        v = transformers.__version__
        if not v.startswith("4."):
            print(f"[CONFLICT] transformers {v} detected (need 4.x). GroundingDINO will break.")
            return False
        print(f"[OK] transformers=={v} (safe)")
        return True
    except ImportError:
        print("[WARN] transformers not installed yet - will be installed by pip.")
        return True

def check_numpy():
    try:
        import numpy
        v = numpy.__version__
        major = int(v.split(".")[0])
        if major >= 2:
            print(f"[CONFLICT] numpy {v} detected (need <2.0). PyTorch may crash.")
            return False
        print(f"[OK] numpy=={v} (safe)")
        return True
    except ImportError:
        print("[WARN] numpy not installed yet - will be installed by pip.")
        return True

ok = check_transformers() and check_numpy()
sys.exit(0 if ok else 1)

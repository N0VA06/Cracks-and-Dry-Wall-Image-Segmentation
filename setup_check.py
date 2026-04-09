"""
setup_check.py
Run this once after installing requirements to verify everything is wired up.

  python setup_check.py
"""

import sys
import os

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

errors = []

# ── Python version ────────────────────────────────────────────────────
v = sys.version_info
print(f"\n{'─'*50}")
print("Environment check")
print(f"{'─'*50}")
label = f"Python {v.major}.{v.minor}.{v.micro}"
if v >= (3, 10):
    print(f"{PASS} {label}")
else:
    print(f"{FAIL} {label}  (3.10+ required for union type hints)")
    errors.append("Python < 3.10")

# ── PyTorch ───────────────────────────────────────────────────────────
try:
    import torch
    cuda = torch.cuda.is_available()
    mps  = torch.backends.mps.is_available()
    gpu  = f"CUDA({torch.cuda.get_device_name(0)})" if cuda else ("MPS" if mps else "CPU only")
    print(f"{PASS} PyTorch {torch.__version__}  |  {gpu}")
except ImportError:
    print(f"{FAIL} PyTorch not installed")
    errors.append("PyTorch missing")

# ── torchvision ───────────────────────────────────────────────────────
try:
    import torchvision
    print(f"{PASS} torchvision {torchvision.__version__}")
except ImportError:
    print(f"{FAIL} torchvision not installed")
    errors.append("torchvision missing")

# ── segment-anything ─────────────────────────────────────────────────
try:
    import segment_anything
    print(f"{PASS} segment-anything (SAM)")
except ImportError:
    print(f"{WARN} segment-anything not installed — SAM model unavailable")
    print(f"       pip install git+https://github.com/facebookresearch/segment-anything.git")

# ── SAM checkpoint ────────────────────────────────────────────────────
try:
    from config import SAM_CHECKPOINT, SAM_MODEL_TYPE
    if os.path.isfile(SAM_CHECKPOINT):
        size_mb = os.path.getsize(SAM_CHECKPOINT) // 1_048_576
        print(f"{PASS} SAM checkpoint ({SAM_MODEL_TYPE}): {SAM_CHECKPOINT}  [{size_mb} MB]")
    else:
        print(f"{WARN} SAM checkpoint not found: {SAM_CHECKPOINT}")
        print(f"       python scripts/download_sam.py --type {SAM_MODEL_TYPE}")
except Exception as e:
    print(f"{WARN} Could not check SAM checkpoint: {e}")

# ── Optional packages ─────────────────────────────────────────────────
for pkg, label in [("albumentations", "albumentations"), ("pycocotools", "pycocotools"), ("matplotlib", "matplotlib")]:
    try:
        __import__(pkg)
        print(f"{PASS} {label}")
    except ImportError:
        print(f"{WARN} {label} not installed (optional)")

# ── Dataset directories ───────────────────────────────────────────────
try:
    from config import DATASET_ROOTS
    print()
    for name, path in DATASET_ROOTS.items():
        if os.path.isdir(path):
            n = sum(1 for f in os.listdir(path) if not f.startswith("."))
            print(f"{PASS} Dataset [{name}]: {path}  ({n} items)")
        else:
            print(f"{WARN} Dataset [{name}] not found: {path}")
            print(f"       Drop your '{os.path.basename(path)}' folder into the 'datasets/' directory.")
except Exception as e:
    print(f"{FAIL} Could not check datasets: {e}")

# ── Output dirs ───────────────────────────────────────────────────────
from config import CHECKPOINT_DIR, LOG_DIR, PREDICTION_DIR
for d in [CHECKPOINT_DIR, LOG_DIR, PREDICTION_DIR]:
    os.makedirs(d, exist_ok=True)
print(f"\n{PASS} Output directories ready")

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
if errors:
    print(f"FAILED  ({len(errors)} error(s)): {', '.join(errors)}")
    sys.exit(1)
else:
    print("All checks passed. You are ready to run main.py")
print(f"{'─'*50}\n")

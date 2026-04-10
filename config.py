"""
config.py — Central configuration for the binary segmentation project.
All hyperparameters, paths, and flags live here.
"""

import os

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_ROOTS = {
    "cracks":  os.path.join(BASE_DIR, "datasets", "cracks.coco"),
    "drywall": os.path.join(BASE_DIR, "datasets", "Drywall"),
}

SPLIT_OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "splits")

OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")
LOG_DIR        = os.path.join(OUTPUT_DIR, "logs")

for _d in [SPLIT_OUTPUT_DIR, CHECKPOINT_DIR, PREDICTION_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────── Label unification ───────────────────────────
FOREGROUND_KEYWORDS = [
    "crack", "cracks",
    "taping_area", "taping", "tape", "taped",
    "joint", "join", "seam", "drywall",
]

# ─────────────────────────── Prompt templates ────────────────────────────
PROMPT_TEMPLATES: dict[str, list[str]] = {
    "crack": [
        "segment crack",
        "segment wall crack",
        "identify crack",
        "find crack",
        "detect crack",
        "highlight crack",
        "show crack",
        "locate crack",
    ],
    "taping": [
        "segment taping area",
        "segment joint tape",
        "segment drywall seam",
        "identify tape joint",
        "find taping area",
        "detect drywall tape",
        "highlight seam",
        "show joint area",
    ],
}

DEFAULT_PROMPT = "segment defect"

ALL_PROMPTS: list[str] = (
    [p for ps in PROMPT_TEMPLATES.values() for p in ps] + [DEFAULT_PROMPT]
)

# ─────────────────────────── Image settings ───────────────────────────
IMAGE_SIZE = (512, 512)
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]

# ─────────────────────────── Training ───────────────────────────
BATCH_SIZE    = 16
NUM_WORKERS   = 4
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0

COSINE_ETA_MIN = 1e-6

BCE_WEIGHT     = 1.0
DICE_WEIGHT    = 1.0
BCE_POS_WEIGHT = 10.0

SAVE_EVERY_N_EPOCHS = 5
KEEP_BEST_ONLY      = True

EARLY_STOPPING_PATIENCE  = 8
EARLY_STOPPING_MIN_DELTA = 1e-4

# ─────────────────────────── Model ───────────────────────────
NUM_CLASSES          = 1
BACKBONE_FREEZE      = True
CUSTOM_BASE_CHANNELS = 32

# ─────────────────────────── Mask quality filtering ──────────────────────────
SKIP_BBOX_MASKS = False

# ─────────────────────────── Dataset augmentation ────────────────────────────
MINORITY_REPEAT = 6

# ─────────────────────────── Text conditioning ───────────────────────────
TEXT_EMBED_DIM  = 128
TEXT_VOCAB_SIZE = 1000
USE_TEXT_PROMPT = True

# ─────────────────────────── Active Learning ───────────────────────────
AL_INITIAL_FRACTION = 0.28
AL_QUERY_FRACTION   = 0.1
AL_ROUNDS           = 5
AL_EPOCHS_PER_ROUND = 10
AL_MC_PASSES        = 10

# ─────────────────────────── Data split ───────────────────────────
SPLIT_RATIOS = (0.80, 0.10, 0.10)
SPLIT_SEED   = 42

# ─────────────────────────── Inference ───────────────────────────
INFERENCE_THRESHOLD = 0.35

# ─────────────────────────── SAM ───────────────────────────────────────
SAM_MODEL_TYPE      = "vit_b"
SAM_CHECKPOINT_FILE = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}
SAM_CHECKPOINT     = os.path.join(CHECKPOINT_DIR, SAM_CHECKPOINT_FILE[SAM_MODEL_TYPE])
SAM_FREEZE_ENCODER = True
SAM_ASPP_CHANNELS  = 128
SAM_LR             = 3e-4
# SAM ViT-B upsamples to 1024x1024 internally — batch=16 needs 12 GB → OOM.
# batch=4 keeps peak memory ~3 GB, fits on A5000.
SAM_BATCH_SIZE     = 4

# ─────────────────────────── Misc ───────────────────────────
SEED   = 42
DEVICE = "cuda"
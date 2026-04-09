"""
config.py — Central configuration for the binary segmentation project.
All hyperparameters, paths, and flags live here.
"""

import os

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_ROOTS = {
    "cracks":  os.path.join(BASE_DIR, "datasets", "cracks.coco"),
    "drywall": os.path.join(BASE_DIR, "datasets", "Drywall-Join-Detect.v2i.coco"),
}

SPLIT_OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "splits")

OUTPUT_DIR        = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR    = os.path.join(OUTPUT_DIR, "checkpoints")
PREDICTION_DIR    = os.path.join(OUTPUT_DIR, "predictions")
LOG_DIR           = os.path.join(OUTPUT_DIR, "logs")

for _d in [SPLIT_OUTPUT_DIR, CHECKPOINT_DIR, PREDICTION_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────── Label unification ───────────────────────────
# Any COCO category whose name contains one of these substrings → foreground (class 1)
FOREGROUND_KEYWORDS = [
    "crack", "cracks",
    "taping_area", "taping", "tape", "taped",
    "joint", "join", "seam", "drywall",
]

# ─────────────────────────── Prompt templates ────────────────────────────
# Used for (a) training-time prompt augmentation
# and (b) building the tokenizer vocabulary.
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

# Flat list of all prompts — used to pre-build tokenizer vocabulary
ALL_PROMPTS: list[str] = (
    [p for ps in PROMPT_TEMPLATES.values() for p in ps] + [DEFAULT_PROMPT]
)

# ─────────────────────────── Image settings ───────────────────────────
IMAGE_SIZE   = (512, 512)   # (H, W) – resize target
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]

# ─────────────────────────── Training ───────────────────────────
BATCH_SIZE        = 8
NUM_WORKERS       = 4
NUM_EPOCHS        = 50
LEARNING_RATE     = 1e-4
WEIGHT_DECAY      = 1e-4
LR_STEP_SIZE      = 15        # StepLR step
LR_GAMMA          = 0.5
GRAD_CLIP         = 1.0       # max gradient norm (0 = disabled)

# Loss weights
BCE_WEIGHT        = 1.0
DICE_WEIGHT       = 1.0

# Checkpoint saving
SAVE_EVERY_N_EPOCHS = 5
KEEP_BEST_ONLY      = True

# ─────────────────────────── Model ───────────────────────────
NUM_CLASSES       = 1         # binary segmentation
BACKBONE_FREEZE   = False     # freeze ResNet backbone in pretrained model

# Custom DeepLabV3 backbone channels
CUSTOM_BASE_CHANNELS = 64

# ─────────────────────────── Text conditioning ───────────────────────────
TEXT_EMBED_DIM    = 128
TEXT_VOCAB_SIZE   = 1000      # simple word-level vocab
USE_TEXT_PROMPT   = True              # text-conditioned binary segmentation

# ─────────────────────────── Active Learning ───────────────────────────
AL_INITIAL_FRACTION = 0.4     # fraction of train set used in round 0
AL_QUERY_FRACTION   = 0.2     # fraction added per round
AL_ROUNDS           = 2
AL_EPOCHS_PER_ROUND = 20
AL_MC_PASSES        = 10      # MC-Dropout forward passes for uncertainty

# ─────────────────────────── Data split ───────────────────────────
SPLIT_RATIOS = (0.80, 0.10, 0.10)   # train / val / test
SPLIT_SEED   = 42

# ─────────────────────────── Inference ───────────────────────────
INFERENCE_THRESHOLD = 0.5     # sigmoid threshold for binary mask

# ─────────────────────────── SAM (Segment Anything Model) ───────────────────────────
SAM_MODEL_TYPE      = "vit_b"          # "vit_b" | "vit_l" | "vit_h"
SAM_CHECKPOINT_FILE = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}
SAM_CHECKPOINT      = os.path.join(
    CHECKPOINT_DIR, SAM_CHECKPOINT_FILE[SAM_MODEL_TYPE]
)
SAM_FREEZE_ENCODER  = True             # freeze ViT encoder, train decoder only
SAM_ASPP_CHANNELS   = 128             # decoder width
SAM_LR              = 5e-4            # slightly higher LR since encoder is frozen

# ─────────────────────────── Misc ───────────────────────────
SEED   = 42
DEVICE = "cuda"               # overridden at runtime if CUDA unavailable

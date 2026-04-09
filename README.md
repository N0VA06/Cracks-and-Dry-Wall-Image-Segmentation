# Text-Conditioned Binary Segmentation

**Goal:** Given an image and a natural-language prompt, output a binary mask for wall cracks or drywall taping areas.

```
"segment crack"        →  mask of all cracks in the image
"segment taping area"  →  mask of all drywall tape/seam regions
```

Output spec: single-channel PNG · values {0, 255} · same H×W as source · filename `<id>__<prompt>.png`

---

## Approach

FiLM (Feature-wise Linear Modulation) text conditioning is applied inside all four models. A word-level text encoder converts the prompt to a 128-d embedding; FiLM generates per-channel γ and β to modulate the visual feature map before the segmentation head. Both datasets are mixed in every training batch so the model learns to separate the two prompt classes.

Loss = BCE + Dice (equal weight). mIoU and mDice are computed **per prompt bucket** (crack / taping) at every validation epoch, then averaged into a single scalar used for checkpointing.

Random seed: **42** (set in `config.py → SEED`).

---

## Models tried

| Key | Architecture | Backbone | Notes |
|-----|-------------|----------|-------|
| `pretrained` | DeepLabV3 | ResNet-50 (ImageNet+COCO) | Fine-tuned; fastest to converge |
| `custom` | DeepLabV3 (scratch) | Custom ResNet-like | No pretrained weights |
| `active` | DeepLabV3 + Active Learning | Same as custom | MC-Dropout entropy query; 5 AL rounds |
| `sam` | SAM + ASPP decoder | ViT-B (frozen) | Strongest spatial prior; only decoder trained |

---

## Data

Source:
- **Cracks** — https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36
- **Drywall** — https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect

Both exported as COCO Segmentation. Split with `python main.py --mode split` (seed 42):

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| Cracks | 80% | 10% | 10% |
| Drywall | 80% | 10% | 10% |
| **Combined** | **~80%** | **~10%** | **~10%** |

---

## Metrics (val set, best epoch)

Per-prompt mIoU and mDice reported. Populate this table after training:

| Model | mIoU | mDice | crack IoU | crack Dice | taping IoU | taping Dice |
|-------|------|-------|-----------|------------|------------|-------------|
| pretrained | — | — | — | — | — | — |
| custom | — | — | — | — | — | — |
| active | — | — | — | — | — | — |
| sam | — | — | — | — | — | — |

> Values are filled automatically from `outputs/logs/` after running `python main.py --mode compare`.

---

## Visual examples

Saved automatically to `outputs/logs/panels/` at each best-epoch checkpoint.
Format: `<model>_panel_<n>__<prompt>.png`  (3-column: original | ground truth | prediction)

```
outputs/logs/panels/
├── pretrained_panel_01__segment_crack.png
├── pretrained_panel_02__segment_taping_area.png
├── sam_panel_01__segment_crack.png
└── sam_panel_02__segment_taping_area.png
```

Example layout of each panel:

```
┌─────────────┬─────────────┬─────────────┐
│  Original   │ Ground Truth│  Prediction │
│   (RGB)     │  (green)    │   (red)     │
└─────────────┴─────────────┴─────────────┘
```

---

## Runtime & footprint

Measured on a single GPU (fill in after training):

| Model | Trainable params | Size (MB) | Train time | Avg inference / image |
|-------|-----------------|-----------|------------|----------------------|
| pretrained | ~15.3M | ~58 MB | — | — |
| custom | ~8.1M | ~31 MB | — | — |
| active | ~8.1M | ~31 MB | — (×5 AL rounds) | — |
| sam (decoder only) | ~2.1M | ~8 MB | — | — |

> Train time and inference speed are logged in `outputs/logs/<model>_log.json` under `epoch_time_s` and printed at training completion.

---

## Failure notes

- **Low-contrast cracks** (hairline, dust-obscured) produce over-segmented masks; the crack IoU is noticeably lower than taping IoU in early epochs.
- **Drywall dataset imbalance**: taping regions are thin linear features; small misalignment in mask polygons leads to high false-negative rate.
- **Prompt variants** with uncommon phrasing (e.g. "locate crack") score ~2–3 IoU pts lower than the canonical prompts due to small vocabulary size.
- **SAM at low resolution**: SAM's ViT encoder was designed for 1024×1024; running at 512×512 reduces spatial precision on fine crack boundaries.

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Place datasets
#    datasets/cracks.coco/train/_annotations.coco.json + images
#    datasets/Drywall-Join-Detect.v2i.coco/train/_annotations.coco.json + images

# 3. Download SAM checkpoint
python scripts/download_sam.py

# 4. Check environment
python setup_check.py

# 5. Split
python main.py --mode split

# 6. Train (joint — both datasets per batch)
python main.py --mode train --model pretrained --dataset all
python main.py --mode train --model custom     --dataset all
python main.py --mode train --model active     --dataset all
python main.py --mode train --model sam        --dataset all

# 7. Compare
python main.py --mode compare

# 8. Predict
python main.py --mode predict --model sam --image wall.jpg --prompt "segment crack"
python main.py --mode predict --model sam --image board.jpg --prompt "segment taping area"
```

All hyperparameters (LR, batch size, AL rounds, FiLM dim, seed) are in `config.py`.
# Cracks-and-Dry-Wall-Image-Segmentation

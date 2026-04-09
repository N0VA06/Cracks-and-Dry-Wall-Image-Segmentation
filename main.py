"""
main.py
Central entry point for the binary segmentation project.

Commands:
  python main.py --mode split
  python main.py --mode train --model pretrained [--dataset cracks|drywall|all]
  python main.py --mode train --model custom
  python main.py --mode train --model active
  python main.py --mode predict --image <path> --prompt "segment crack"
  python main.py --mode compare
"""

import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from config import (
    BATCH_SIZE, NUM_WORKERS, DEVICE, SEED,
    SPLIT_OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR,
    NUM_CLASSES, USE_TEXT_PROMPT,
    TEXT_EMBED_DIM, TEXT_VOCAB_SIZE,
    CUSTOM_BASE_CHANNELS,
    BACKBONE_FREEZE,
    AL_INITIAL_FRACTION, AL_QUERY_FRACTION, AL_ROUNDS, AL_EPOCHS_PER_ROUND,
    SAM_MODEL_TYPE, SAM_CHECKPOINT, SAM_FREEZE_ENCODER, SAM_ASPP_CHANNELS, SAM_LR,
)


# ══════════════════════════════════════════════════════
#  Reproducibility
# ══════════════════════════════════════════════════════

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════
#  Device
# ══════════════════════════════════════════════════════

def get_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("[Device] Apple MPS")
    else:
        d = torch.device("cpu")
        print("[Device] CPU (training may be slow)")
    return d


# ══════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════

def resolve_split_paths(dataset_name: str) -> dict[str, tuple[str, str]]:
    """
    Returns {subset: (json_path, image_dir)} for train/val/test.

    Split JSONs live in datasets/splits/<name>/ but images remain in
    the original dataset folder. image_dir always points to the
    original image folder, never the splits folder.
    """
    paths     = {}
    split_dir = os.path.join(SPLIT_OUTPUT_DIR, dataset_name)
    root      = config.DATASET_ROOTS.get(dataset_name, "")

    # Find original image directory once
    orig_img_dir = None
    for candidate in ["train", "valid", "images"]:
        d = os.path.join(root, candidate)
        if os.path.isdir(d):
            orig_img_dir = d
            break

    for subset in ("train", "val", "test"):
        json_path = os.path.join(split_dir, f"{dataset_name}_{subset}.json")

        if os.path.isfile(json_path):
            # Split file found — images are in the original dataset folder
            img_dir = orig_img_dir or os.path.join(root, "train")
        else:
            # No split yet — fall back to raw dataset annotation file
            img_dir   = orig_img_dir or ""
            json_path = ""
            for folder in (["train"] if subset == "train" else ["valid", "val", subset]):
                d = os.path.join(root, folder)
                for name in ["_annotations.coco.json", "instances_default.json", "annotations.json"]:
                    p = os.path.join(d, name)
                    if os.path.isfile(p):
                        json_path = p
                        img_dir   = d
                        break
                if json_path:
                    break

        paths[subset] = (json_path, img_dir)

    return paths


def build_loaders(
    dataset_name: str,
    tokenizer=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    from preprocessing.coco_dataset import build_dataset
    from preprocessing.transforms import get_train_transforms, get_val_transforms

    split_paths = resolve_split_paths(dataset_name)

    def _loader(subset, shuffle):
        json_p, img_dir = split_paths[subset]
        if not os.path.isfile(json_p):
            raise FileNotFoundError(
                f"Annotation file not found: {json_p}\n"
                f"Run  python main.py --mode split  first."
            )
        is_train = (subset == "train")
        tf  = get_train_transforms() if is_train else get_val_transforms()
        ds  = build_dataset(json_p, img_dir, transform=tf, tokenizer=tokenizer, train=is_train)
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=shuffle and len(ds) > BATCH_SIZE,
        )

    return _loader("train", True), _loader("val", False), _loader("test", False)


def build_tokenizer_and_encoder(device):
    from models.text_conditioning import SimpleTokenizer, TextEncoder
    from config import ALL_PROMPTS, TEXT_VOCAB_SIZE, TEXT_EMBED_DIM

    tokenizer = SimpleTokenizer(vocab_size=TEXT_VOCAB_SIZE)
    # ── CRITICAL: build vocabulary from all known prompt templates ──────
    # Without this every word maps to <unk> and FiLM gets random noise.
    tokenizer.build_vocab(ALL_PROMPTS)
    print(f"[Tokenizer] Vocab built — {len(tokenizer.word2idx)} tokens from {len(ALL_PROMPTS)} prompt templates")

    text_encoder = TextEncoder(vocab_size=TEXT_VOCAB_SIZE, embed_dim=TEXT_EMBED_DIM)
    text_encoder.to(device)
    return tokenizer, text_encoder


# ══════════════════════════════════════════════════════
#  Model factories
# ══════════════════════════════════════════════════════

def make_pretrained_model():
    from models.pretrained_deeplabv3 import PretrainedDeepLabV3
    return PretrainedDeepLabV3(
        backbone="resnet50",
        freeze_backbone=BACKBONE_FREEZE,
        use_text_prompt=USE_TEXT_PROMPT,
        text_embed_dim=TEXT_EMBED_DIM,
    )


def make_custom_model(use_dropout: bool = False):
    from models.custom_deeplabv3 import CustomDeepLabV3
    return CustomDeepLabV3(
        base_channels=CUSTOM_BASE_CHANNELS,
        num_classes=NUM_CLASSES,
        use_dropout=use_dropout,
        use_text_prompt=USE_TEXT_PROMPT,
        text_embed_dim=TEXT_EMBED_DIM,
    )


def make_sam_model():
    from models.sam_segmentor import SAMSegmentor
    if not os.path.isfile(SAM_CHECKPOINT):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {SAM_CHECKPOINT}\n"
            f"Run: python scripts/download_sam.py --type {SAM_MODEL_TYPE}"
        )
    return SAMSegmentor(
        sam_checkpoint=SAM_CHECKPOINT,
        model_type=SAM_MODEL_TYPE,
        freeze_encoder=SAM_FREEZE_ENCODER,
        aspp_channels=SAM_ASPP_CHANNELS,
        use_text_prompt=USE_TEXT_PROMPT,
        text_embed_dim=TEXT_EMBED_DIM,
    )


# ══════════════════════════════════════════════════════
#  Mode: split
# ══════════════════════════════════════════════════════

def mode_split():
    from preprocessing.split_dataset import split_all_datasets
    split_all_datasets()


# ══════════════════════════════════════════════════════
#  Mode: train
# ══════════════════════════════════════════════════════

def mode_train(args, device):
    from training.train import train as run_train

    dataset_name = getattr(args, "dataset", "cracks") or "cracks"
    print(f"\n[Train] Dataset: {dataset_name}  |  Model: {args.model}")

    # ── Tokenizer + text encoder (always on — text conditioning is the goal) ──
    tokenizer, text_encoder = build_tokenizer_and_encoder(device)

    # ── DataLoaders ───────────────────────────────────────────────────
    if dataset_name == "all":
        # Combined loader: model sees BOTH "segment crack" and "segment taping area"
        from preprocessing.combined_dataset import build_combined_loaders
        print("\n[Train] Combined dataset mode — both datasets mixed per batch")
        train_loader, val_loader, _ = build_combined_loaders(tokenizer)
        if train_loader is None:
            raise RuntimeError("No training data found. Run --mode split first.")
    else:
        train_loader, val_loader, _ = build_loaders(dataset_name, tokenizer)

    if args.model == "pretrained":
        model = make_pretrained_model()
        run_train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            model_tag="pretrained",
            text_encoder=text_encoder,
            use_text_prompt=True,
        )

    elif args.model == "custom":
        model = make_custom_model(use_dropout=False)
        run_train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            model_tag="custom",
            text_encoder=text_encoder,
            use_text_prompt=True,
        )

    elif args.model == "active":
        from active_learning.loop import active_learning_loop
        from preprocessing.coco_dataset import build_dataset
        from preprocessing.transforms import get_train_transforms

        if dataset_name == "all":
            from torch.utils.data import ConcatDataset
            from preprocessing.combined_dataset import _find_json
            train_sets = []
            for ds_name in config.DATASET_ROOTS:
                result = _find_json(ds_name, "train")
                if result:
                    json_p, img_dir = result
                    train_sets.append(build_dataset(json_p, img_dir,
                                                    transform=get_train_transforms(),
                                                    tokenizer=tokenizer, train=True))
            full_train_ds = ConcatDataset(train_sets)
        else:
            split_paths   = resolve_split_paths(dataset_name)
            json_p, img_dir = split_paths["train"]
            full_train_ds = build_dataset(json_p, img_dir,
                                          transform=get_train_transforms(),
                                          tokenizer=tokenizer, train=True)

        active_learning_loop(
            model_factory=lambda: make_custom_model(use_dropout=True),
            train_dataset=full_train_ds,
            val_loader=val_loader,
            device=device,
            model_tag="active",
            text_encoder=text_encoder,
            use_text_prompt=True,
        )

    elif args.model == "sam":
        model = make_sam_model()
        run_train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            model_tag="sam",
            text_encoder=text_encoder,
            use_text_prompt=True,
            learning_rate=SAM_LR,
            trainable_params=model.get_trainable_params(),
        )

    else:
        raise ValueError(f"Unknown model type: {args.model}")


# ══════════════════════════════════════════════════════
#  Mode: predict
# ══════════════════════════════════════════════════════

def mode_predict(args, device):
    from inference.predict import predict

    model_map = {
        "pretrained": (make_pretrained_model,              "pretrained_best.pth"),
        "custom"    : (lambda: make_custom_model(False),   "custom_best.pth"),
        "active"    : (lambda: make_custom_model(True),    "active_round5_best.pth"),
        "sam"       : (make_sam_model,                     "sam_best.pth"),
    }

    model_type = getattr(args, "model", "pretrained") or "pretrained"
    factory, ckpt_name = model_map[model_type]
    model         = factory()
    ckpt_path     = os.path.join(CHECKPOINT_DIR, ckpt_name)

    tokenizer, text_encoder = (None, None)
    if USE_TEXT_PROMPT:
        tokenizer, text_encoder = build_tokenizer_and_encoder(device)

    predict(
        model=model,
        image_path=args.image,
        prompt=args.prompt,
        device=device,
        checkpoint_path=ckpt_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        use_text_prompt=USE_TEXT_PROMPT,
    )


# ══════════════════════════════════════════════════════
#  Mode: compare
# ══════════════════════════════════════════════════════

def mode_compare():
    from utils.visualization import plot_model_comparison, plot_training_curve

    model_tags   = ["pretrained", "custom", "active", "sam"]
    metrics_dict = {}

    for tag in model_tags:
        log_path = os.path.join(LOG_DIR, f"{tag}_log.json")
        if not os.path.isfile(log_path):
            # Try AL summary
            log_path = os.path.join(LOG_DIR, f"{tag}_al_summary.json")

        if not os.path.isfile(log_path):
            print(f"[Compare] No log found for model: {tag}")
            continue

        with open(log_path) as f:
            history = json.load(f)

        # Extract best val metrics
        if isinstance(history, list) and history and "val" in history[0]:
            best = max(history, key=lambda r: r["val"]["iou"])
            metrics_dict[tag] = best["val"]
        elif isinstance(history, list) and history and "best_iou" in history[0]:
            # AL summary
            best_round = max(history, key=lambda r: r["best_iou"])
            metrics_dict[tag] = {"iou": best_round["best_iou"], "dice": 0.0}

        # Also plot individual curves
        full_log = os.path.join(LOG_DIR, f"{tag}_log.json")
        if os.path.isfile(full_log):
            plot_training_curve(full_log)

    if metrics_dict:
        save_path = os.path.join(LOG_DIR, "model_comparison.png")
        plot_model_comparison(metrics_dict, save_path)

        print("\nFinal Comparison:")
        print("-" * 50)
        for m, vals in metrics_dict.items():
            print(f"  {m:20s} → IoU: {vals.get('iou',0):.4f} | Dice: {vals.get('dice',0):.4f}")
    else:
        print("[Compare] No model logs found. Train models first.")


# ══════════════════════════════════════════════════════
#  Argument parser
# ══════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Binary Segmentation Pipeline")
    p.add_argument("--mode",    required=True,
                   choices=["split", "train", "predict", "compare"],
                   help="Operation mode")
    p.add_argument("--model",   default="pretrained",
                   choices=["pretrained", "custom", "active", "sam"],
                   help="Model type (for train/predict)")
    p.add_argument("--dataset", default="cracks",
                   choices=list(config.DATASET_ROOTS.keys()) + ["all"],
                   help="Dataset to use")
    p.add_argument("--image",   default=None,
                   help="Path to input image (predict mode)")
    p.add_argument("--prompt",  default="segment defect",
                   help="Text prompt (predict mode)")
    p.add_argument("--checkpoint", default=None,
                   help="Override checkpoint path for prediction")
    return p.parse_args()


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════

def main():
    set_seed(SEED)
    args   = parse_args()
    device = get_device()

    print(f"\n{'='*60}")
    print(f"  Mode: {args.mode}  |  Model: {args.model}")
    print(f"{'='*60}\n")

    if args.mode == "split":
        mode_split()

    elif args.mode == "train":
        if args.dataset == "all":
            for ds_name in config.DATASET_ROOTS:
                args.dataset = ds_name
                mode_train(args, device)
        else:
            mode_train(args, device)

    elif args.mode == "predict":
        if not args.image:
            raise ValueError("--image is required for predict mode.")
        mode_predict(args, device)

    elif args.mode == "compare":
        mode_compare()


if __name__ == "__main__":
    main()

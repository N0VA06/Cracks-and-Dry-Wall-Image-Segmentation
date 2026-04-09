"""
training/train.py
Full training loop with per-prompt mIoU / mDice logging.

Per-epoch console output:
  Epoch 003/050 | Loss 0.3821 | mIoU 0.6934 | mDice 0.8102 | LR 1.00e-04 | 38.4s
    crack   → IoU 0.7421  Dice 0.8534  [n=240]
    taping  → IoU 0.6447  Dice 0.7670  [n=160]

JSON log entry:
  {epoch, train:{loss,bce,dice}, val:{loss,miou,mdice,overall:{...},crack:{...},taping:{...}}, lr}
"""

import os
import json
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from config import (
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    LR_STEP_SIZE, LR_GAMMA, GRAD_CLIP,
    BCE_WEIGHT, DICE_WEIGHT,
    SAVE_EVERY_N_EPOCHS, KEEP_BEST_ONLY,
    CHECKPOINT_DIR, LOG_DIR,
)
from training.loss import SegmentationLoss
from training.metrics import format_metrics_log
from training.validate import validate


# ══════════════════════════════════════════════════════
#  Model footprint helper
# ══════════════════════════════════════════════════════

def model_footprint(model: torch.nn.Module) -> dict:
    """Returns parameter counts and approximate size in MB."""
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb   = total * 4 / 1_048_576          # float32
    return {
        "total_params"    : total,
        "trainable_params": trainable,
        "size_mb"         : round(size_mb, 1),
    }


# ══════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════

def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    model_tag: str = "model",
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    text_encoder=None,
    use_text_prompt: bool = True,
    resume_checkpoint: str | None = None,
    trainable_params: list | None = None,
) -> dict:
    """
    Train model and return best-epoch summary dict.

    Logged per epoch (console + JSON):
      train: loss, bce, dice
      val:   loss, miou, mdice, per-prompt IoU/Dice (crack & taping)
    """
    model.to(device)
    if text_encoder is not None:
        text_encoder.to(device)

    fp = model_footprint(model)
    print(
        f"\n[{model_tag}] Parameters: {fp['total_params']:,}  "
        f"Trainable: {fp['trainable_params']:,}  "
        f"Size: {fp['size_mb']} MB"
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────
    params_to_optimize = trainable_params if trainable_params is not None else model.parameters()
    optimizer = AdamW(params_to_optimize, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    criterion = SegmentationLoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT)

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    best_miou   = 0.0
    history     = []

    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        history     = ckpt.get("history", [])
        print(f"[Resume] epoch {start_epoch}, best mIoU {best_miou:.4f}")

    log_path    = os.path.join(LOG_DIR, f"{model_tag}_log.json")
    total_train_time = 0.0

    # ── Epoch loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        if text_encoder is not None:
            text_encoder.train()

        epoch_loss = epoch_bce = epoch_dice = 0.0
        t0 = time.time()

        for batch_idx, (images, masks, token_ids, _) in enumerate(train_loader):
            images    = images.to(device, non_blocking=True)
            masks     = masks.to(device, non_blocking=True)
            token_ids = token_ids.to(device, non_blocking=True)

            text_embeds = None
            if use_text_prompt and text_encoder is not None:
                text_embeds = text_encoder(token_ids)

            optimizer.zero_grad()
            output = model(images, text_embeds)
            logits = output["out"]

            loss, sub = criterion(logits, masks)
            if "aux" in output:
                aux_loss, _ = criterion(output["aux"], masks)
                loss = loss + 0.4 * aux_loss

            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bce  += sub["bce"]
            epoch_dice += sub["dice"]

            # Progress print every ~20% of epoch
            log_every = max(1, len(train_loader) // 5)
            if (batch_idx + 1) % log_every == 0:
                print(
                    f"  [{model_tag}] Epoch {epoch+1} "
                    f"batch {batch_idx+1}/{len(train_loader)} "
                    f"loss {loss.item():.4f}"
                )

        scheduler.step()
        elapsed = time.time() - t0
        total_train_time += elapsed

        n = max(len(train_loader), 1)
        train_metrics = {
            "loss": round(epoch_loss / n, 4),
            "bce" : round(epoch_bce  / n, 4),
            "dice": round(epoch_dice / n, 4),
        }

        # ── Validation with per-prompt metrics ───────────────────────
        val_metrics = validate(
            model, val_loader, criterion, device,
            use_text_prompt=use_text_prompt,
            text_encoder=text_encoder,
        )

        miou  = val_metrics.get("miou",  val_metrics["overall"]["iou"])
        mdice = val_metrics.get("mdice", val_metrics["overall"]["dice"])

        # ── Console output ────────────────────────────────────────────
        print(
            f"\nEpoch {epoch+1:03d}/{start_epoch+num_epochs:03d} | "
            f"Loss {train_metrics['loss']:.4f} | "
            f"mIoU {miou:.4f} | "
            f"mDice {mdice:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )
        print(format_metrics_log(val_metrics))

        # ── JSON log ──────────────────────────────────────────────────
        record = {
            "epoch"      : epoch + 1,
            "train"      : train_metrics,
            "val"        : val_metrics,
            "lr"         : scheduler.get_last_lr()[0],
            "epoch_time_s": round(elapsed, 1),
        }
        history.append(record)

        # ── Checkpoint ────────────────────────────────────────────────
        is_best = miou > best_miou
        if is_best:
            best_miou = miou

        save_this = is_best or (
            not KEEP_BEST_ONLY and (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0
        )
        if save_this:
            ckpt_name = (
                f"{model_tag}_best.pth" if is_best
                else f"{model_tag}_epoch{epoch+1}.pth"
            )
            ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_miou"      : best_miou,
                "val_metrics"    : val_metrics,
                "history"        : history,
                "footprint"      : fp,
            }, ckpt_path)
            if is_best:
                print(f"  ✓ Best mIoU {best_miou:.4f} → {ckpt_path}")
                # Save orig | GT | pred visual panels for this best epoch
                try:
                    from utils.visualization import save_val_panels
                    panel_dir = os.path.join(LOG_DIR, "panels")
                    save_val_panels(
                        model=model,
                        loader=val_loader,
                        device=device,
                        save_dir=panel_dir,
                        model_tag=model_tag,
                        n_panels=4,
                        text_encoder=text_encoder,
                    )
                except Exception as e:
                    print(f"  [panels] skipped: {e}")

        # Write log after every epoch
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[{model_tag}] Training complete")
    print(f"  Best val mIoU : {best_miou:.4f}")
    print(f"  Total time    : {total_train_time/60:.1f} min")
    print(f"  Avg per epoch : {total_train_time/max(num_epochs,1):.1f}s")
    print(f"  Log           : {log_path}")
    print(f"{'─'*60}\n")

    return {
        "best_miou"        : best_miou,
        "history"          : history,
        "total_train_time" : total_train_time,
        "footprint"        : fp,
    }
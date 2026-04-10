"""
training/train.py
Training loop compatible with all 4 models: pretrained, custom, active, sam.

Features:
  • tqdm epoch bar  — persistent, shows live mIoU/mDice/LR after each epoch
  • tqdm batch bar  — transient (leave=False), shows live loss/bce/dice per step
  • Python logging  — file (outputs/logs/<model>.log) + console via tqdm.write
  • CosineAnnealingLR scheduler
  • Early stopping
  • Per-prompt mIoU/mDice (crack | taping) logged every epoch
"""

import os
import json
import time
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import (
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    COSINE_ETA_MIN, GRAD_CLIP,
    BCE_WEIGHT, DICE_WEIGHT,
    SAVE_EVERY_N_EPOCHS, KEEP_BEST_ONLY,
    CHECKPOINT_DIR, LOG_DIR,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
)
from training.loss import SegmentationLoss
from training.metrics import format_metrics_log
from training.validate import validate


# ══════════════════════════════════════════════════════
#  Logger
# ══════════════════════════════════════════════════════

def _build_logger(model_tag: str) -> logging.Logger:
    """
    Logger that writes to file AND console without breaking tqdm bars.
    Console output uses tqdm.write() so bars stay intact.
    """
    logger = logging.getLogger(f"train.{model_tag}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S")

    os.makedirs(LOG_DIR, exist_ok=True)
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"{model_tag}_train.log"), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    class TqdmHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                tqdm.write(self.format(record))
            except Exception:
                self.handleError(record)

    ch = TqdmHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


# ══════════════════════════════════════════════════════
#  Footprint
# ══════════════════════════════════════════════════════

def model_footprint(model: torch.nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params"    : total,
        "trainable_params": trainable,
        "size_mb"         : round(total * 4 / 1_048_576, 1),
    }


# ══════════════════════════════════════════════════════
#  Main training function
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
    Train any of the 4 segmentation models.

    Args:
        trainable_params: if set (e.g. SAM decoder only), only these params
                          are passed to the optimizer. Frozen params are skipped.
    Returns:
        {"best_miou", "history", "total_train_time", "footprint"}
    """
    log = _build_logger(model_tag)

    model.to(device)
    if text_encoder is not None:
        text_encoder.to(device)

    fp = model_footprint(model)
    log.info(
        f"[{model_tag}] params={fp['total_params']:,}  "
        f"trainable={fp['trainable_params']:,}  "
        f"size={fp['size_mb']} MB"
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────
    opt_params = trainable_params if trainable_params is not None else model.parameters()
    optimizer  = AdamW(opt_params, lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler  = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=COSINE_ETA_MIN)
    criterion  = SegmentationLoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT)
    criterion.to(device)

    # ── Resume from checkpoint ────────────────────────────────────────
    start_epoch    = 0
    best_miou      = 0.0
    history: list  = []
    no_improve_cnt = 0

    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        history     = ckpt.get("history", [])
        log.info(f"Resumed from epoch {start_epoch}, best mIoU {best_miou:.4f}")

    log_path         = os.path.join(LOG_DIR, f"{model_tag}_log.json")
    total_train_time = 0.0
    end_epoch        = start_epoch + num_epochs

    # ── Epoch bar (persistent across all epochs) ──────────────────────
    epoch_bar = tqdm(
        range(start_epoch, end_epoch),
        desc=f"{model_tag}",
        unit="ep",
        dynamic_ncols=True,
        colour="cyan",
    )

    for epoch in epoch_bar:
        model.train()
        if text_encoder is not None:
            text_encoder.train()

        epoch_loss = epoch_bce = epoch_dice = 0.0
        t0 = time.time()

        # ── Batch bar (disappears after each epoch) ───────────────────
        batch_bar = tqdm(
            train_loader,
            desc=f"  ep {epoch+1:02d}/{end_epoch:02d}",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
            colour="green",
        )

        for images, masks, token_ids, _ in batch_bar:
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

            batch_bar.set_postfix(
                loss=f"{loss.item():.3f}",
                bce=f"{sub['bce']:.3f}",
                dice=f"{sub['dice']:.3f}",
            )

        batch_bar.close()
        scheduler.step()   # step cosine once per epoch

        elapsed = time.time() - t0
        total_train_time += elapsed

        n = max(len(train_loader), 1)
        train_metrics = {
            "loss": round(epoch_loss / n, 4),
            "bce" : round(epoch_bce  / n, 4),
            "dice": round(epoch_dice / n, 4),
        }

        # ── Validation ───────────────────────────────────────────────
        val_metrics = validate(
            model, val_loader, criterion, device,
            use_text_prompt=use_text_prompt,
            text_encoder=text_encoder,
        )

        current_lr = scheduler.get_last_lr()[0]
        miou  = val_metrics.get("miou",  val_metrics["overall"]["iou"])
        mdice = val_metrics.get("mdice", val_metrics["overall"]["dice"])
        is_best = miou > best_miou

        # Update persistent epoch bar
        epoch_bar.set_postfix(
            loss=f"{train_metrics['loss']:.3f}",
            mIoU=f"{miou:.4f}",
            mDice=f"{mdice:.4f}",
            LR=f"{current_lr:.1e}",
            best=f"{max(best_miou, miou):.4f}",
        )

        # Log to file + console
        log.info(
            f"Epoch {epoch+1:03d}/{end_epoch:03d} | "
            f"loss {train_metrics['loss']:.4f} | "
            f"mIoU {miou:.4f} | mDice {mdice:.4f} | "
            f"LR {current_lr:.2e} | {elapsed:.1f}s"
            + (" ← best" if is_best else "")
        )
        log.info(format_metrics_log(val_metrics))

        if is_best:
            best_miou = miou

        history.append({
            "epoch"       : epoch + 1,
            "train"       : train_metrics,
            "val"         : val_metrics,
            "lr"          : current_lr,
            "epoch_time_s": round(elapsed, 1),
            "is_best"     : is_best,
        })

        # ── Checkpoint ───────────────────────────────────────────────
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
                "scheduler_state": scheduler.state_dict(),
                "best_miou"      : best_miou,
                "val_metrics"    : val_metrics,
                "history"        : history,
                "footprint"      : fp,
            }, ckpt_path)
            log.info(f"Checkpoint → {ckpt_name}")

            if is_best:
                try:
                    from utils.visualization import save_val_panels
                    save_val_panels(
                        model=model, loader=val_loader, device=device,
                        save_dir=os.path.join(LOG_DIR, "panels"),
                        model_tag=model_tag, n_panels=4,
                        text_encoder=text_encoder,
                    )
                except Exception as e:
                    log.debug(f"Panels skipped: {e}")

        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

        # ── Early stopping ────────────────────────────────────────────
        if miou >= best_miou - EARLY_STOPPING_MIN_DELTA:
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= EARLY_STOPPING_PATIENCE:
                log.warning(
                    f"Early stopping: no mIoU gain for "
                    f"{EARLY_STOPPING_PATIENCE} epochs (stopped at {epoch+1})"
                )
                break

    epoch_bar.close()

    log.info("=" * 55)
    log.info(f"Training complete — {model_tag}")
    log.info(f"  Best val mIoU : {best_miou:.4f}")
    log.info(f"  Total time    : {total_train_time/60:.1f} min")
    log.info(f"  Avg per epoch : {total_train_time/max(num_epochs,1):.1f}s")
    log.info(f"  JSON log      : {log_path}")
    log.info("=" * 55)

    return {
        "best_miou"       : best_miou,
        "history"         : history,
        "total_train_time": total_train_time,
        "footprint"       : fp,
    }
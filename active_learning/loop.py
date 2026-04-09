"""
active_learning/loop.py
Orchestrates the active learning cycle:
  Round 0 : train on initial labelled subset
  Round k  : compute uncertainty on unlabelled pool
             → query top-K samples
             → expand labelled pool
             → continue training
"""

import os
import json
import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, NUM_WORKERS,
    AL_ROUNDS, AL_EPOCHS_PER_ROUND, AL_MC_PASSES,
    AL_INITIAL_FRACTION, AL_QUERY_FRACTION,
    CHECKPOINT_DIR, LOG_DIR,
    USE_TEXT_PROMPT,
)
from active_learning.sampler import ActiveLearningSampler
from active_learning.uncertainty import mc_dropout_uncertainty
from training.train import train as run_train


def active_learning_loop(
    model_factory,         # callable () → nn.Module with MC-Dropout
    train_dataset,
    val_loader: DataLoader,
    device: torch.device,
    model_tag: str = "active",
    text_encoder=None,
    use_text_prompt: bool = USE_TEXT_PROMPT,
):
    """
    Full active learning loop.

    Args:
        model_factory : callable that returns a fresh (or reused) model instance.
                        Called once; weights persist across rounds.
        train_dataset : full labelled training dataset (all samples)
        val_loader    : fixed validation DataLoader
        device        : torch.device
        model_tag     : prefix for checkpoint / log filenames
    """
    sampler = ActiveLearningSampler(
        dataset=train_dataset,
        initial_fraction=AL_INITIAL_FRACTION,
        query_fraction=AL_QUERY_FRACTION,
    )

    model = model_factory()
    model.to(device)

    all_round_metrics = []
    best_checkpoint   = os.path.join(CHECKPOINT_DIR, f"{model_tag}_best.pth")

    for al_round in range(AL_ROUNDS):
        print("\n" + "=" * 60)
        print(f"  Active Learning Round {al_round + 1}/{AL_ROUNDS}")
        print(f"  Labelled pool size: {sampler.n_labelled}")
        print("=" * 60)

        # ── Build train loader from labelled subset ───────────────────
        labelled_subset = sampler.get_labelled_subset()
        train_loader    = DataLoader(
            labelled_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=len(labelled_subset) > BATCH_SIZE,
        )

        # ── Train for N epochs ────────────────────────────────────────
        round_tag = f"{model_tag}_round{al_round + 1}"
        results   = run_train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            model_tag=round_tag,
            num_epochs=AL_EPOCHS_PER_ROUND,
            text_encoder=text_encoder,
            use_text_prompt=use_text_prompt,
            # Resume from previous round's best checkpoint (if exists)
            resume_checkpoint=best_checkpoint if al_round > 0 and os.path.isfile(best_checkpoint) else None,
        )

        all_round_metrics.append({
            "round"          : al_round + 1,
            "n_labelled"     : sampler.n_labelled,
            "best_iou"       : results["best_iou"],
        })

        # ── Stop if unlabelled pool is empty ─────────────────────────
        if sampler.n_unlabelled == 0:
            print("Unlabelled pool exhausted — stopping AL loop.")
            break

        # ── Compute uncertainty on unlabelled pool ───────────────────
        unlabelled_subset  = sampler.get_unlabelled_subset()
        unlabelled_loader  = DataLoader(
            unlabelled_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        print(f"\nComputing uncertainty over {sampler.n_unlabelled} unlabelled samples ...")
        scores = mc_dropout_uncertainty(
            model=model,
            loader=unlabelled_loader,
            device=device,
            n_passes=AL_MC_PASSES,
            use_text_prompt=use_text_prompt,
            text_encoder=text_encoder,
        )

        # ── Query + expand labelled set ───────────────────────────────
        selected = sampler.query(scores)
        sampler.expand_labelled(selected)

    # ── Save final AL summary log ─────────────────────────────────────
    log_path = os.path.join(LOG_DIR, f"{model_tag}_al_summary.json")
    with open(log_path, "w") as f:
        json.dump(all_round_metrics, f, indent=2)

    print(f"\nActive learning complete. Summary saved: {log_path}")
    for r in all_round_metrics:
        print(f"  Round {r['round']:2d} | Labelled {r['n_labelled']:5d} | Best IoU {r['best_iou']:.4f}")

    return all_round_metrics

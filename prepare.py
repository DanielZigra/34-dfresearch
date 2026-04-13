#!/usr/bin/env python3
"""
prepare.py — Data preparation and evaluation utilities for dfresearch.

THIS FILE IS READ-ONLY. Do not modify it during autoresearch experiments.
It contains fixed evaluation metrics, data downloading, and constants that
ensure experiments are fairly comparable.

Usage:
    uv run prepare.py                         # Download all modalities
    uv run prepare.py --modality image        # Download image datasets only
    uv run prepare.py --modality video        # Download video datasets only
    uv run prepare.py --modality audio        # Download audio datasets only
    uv run prepare.py --modality image_human  # Human-face image datasets (gasbench + local overrides)
"""

import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef, brier_score_loss, accuracy_score

# ──────────────────────────────────────────────────────────────────────────────
# Constants (DO NOT MODIFY)
# ──────────────────────────────────────────────────────────────────────────────

TIME_BUDGET = 600             # 10-minute training budget (wall clock)
TARGET_IMAGE_SIZE = (224, 224)
TARGET_VIDEO_SIZE = (224, 224)
NUM_VIDEO_FRAMES = 16
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 6.0          # seconds
AUDIO_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)  # 96000

DEFAULT_IMAGE_BATCH_SIZE = 32
DEFAULT_VIDEO_BATCH_SIZE = 4
DEFAULT_AUDIO_BATCH_SIZE = 16

LABEL_REAL = 0
LABEL_SYNTHETIC = 1


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation metrics (DO NOT MODIFY)
# ──────────────────────────────────────────────────────────────────────────────

def compute_sn34_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alpha: float = 1.2,
    beta: float = 1.8,
) -> dict:
    """
    Compute the sn34_score used by BitMind Subnet 34.

    sn34_score = sqrt(MCC_norm^alpha * Brier_norm^beta)

    where:
        MCC_norm = (MCC + 1) / 2        (normalized to [0, 1])
        Brier_norm = 1 - brier_score    (higher is better)

    Args:
        y_true: Ground truth labels (0 = real, 1 = synthetic).
        y_prob: Predicted probability of synthetic class.
        alpha: Exponent for MCC component (default: 1.2).
        beta: Exponent for Brier component (default: 1.8).

    Returns:
        Dictionary with sn34_score, mcc, brier, accuracy, and component scores.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return {
            "sn34_score": 0.0, "mcc": 0.0, "mcc_norm": 0.5,
            "brier": 0.25, "brier_norm": 0.75, "accuracy": 0.0,
        }

    y_pred = (y_prob >= 0.5).astype(int)

    # MCC is undefined when only one class is present; fall back to 0
    unique_labels = np.unique(y_true)
    unique_preds = np.unique(y_pred)
    if len(unique_labels) < 2 or len(unique_preds) < 2:
        mcc = 0.0
    else:
        mcc = float(matthews_corrcoef(y_true, y_pred))
        if np.isnan(mcc):
            mcc = 0.0

    brier = float(brier_score_loss(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))

    mcc_norm = (mcc + 1.0) / 2.0
    brier_norm = 1.0 - brier

    mcc_norm = max(mcc_norm, 1e-10)
    brier_norm = max(brier_norm, 1e-10)

    sn34 = (mcc_norm ** alpha * brier_norm ** beta) ** 0.5

    return {
        "sn34_score": float(sn34),
        "mcc": mcc,
        "mcc_norm": float(mcc_norm),
        "brier": brier,
        "brier_norm": float(brier_norm),
        "accuracy": acc,
    }


def evaluate_model(model, dataloader, device="cuda") -> dict:
    """
    Evaluate a model on a dataloader and return sn34 metrics.

    The model should output [B, 2] logits. We apply softmax to get
    probabilities and use p(synthetic) for scoring.

    Args:
        model: PyTorch model producing [B, 2] logits.
        dataloader: DataLoader yielding (inputs, labels) batches.
        device: Device string.

    Returns:
        Dictionary of evaluation metrics.
    """
    import torch

    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(device)
            logits = model(batch_inputs)  # [B, 2]
            probs = torch.softmax(logits.float(), dim=-1)
            p_synthetic = probs[:, 1].cpu().numpy()

            all_labels.extend(batch_labels.cpu().numpy().tolist())
            all_probs.extend(p_synthetic.tolist())

    if len(all_labels) == 0:
        print("WARNING: No validation samples found. Returning zero scores.")
        return compute_sn34_score(np.array([]), np.array([]))

    return compute_sn34_score(np.array(all_labels), np.array(all_probs))


# ──────────────────────────────────────────────────────────────────────────────
# Data download
# ──────────────────────────────────────────────────────────────────────────────

def download_datasets(modality: str, max_workers: int = 4, max_samples: int = 500):
    """Download and cache all datasets for a modality using concurrent workers."""
    from dfresearch.data import download_all_datasets
    download_all_datasets(
        modality, max_workers=max_workers,
        max_samples_per_dataset=max_samples, progress=True,
    )


def verify_cache(modality: str):
    """Print cache status for all datasets of a modality."""
    from dfresearch.data import get_cache_dir, load_dataset_config, _count_media_files

    config = load_dataset_config(modality)
    total_samples = 0
    real_count = 0
    fake_count = 0

    print(f"\nCache status for {modality} datasets:")
    print(f"{'Dataset':<35} {'Type':<12} {'Samples':>8} {'Status'}")
    print("-" * 70)

    for ds_cfg in config["datasets"]:
        name = ds_cfg["name"]
        media_type = ds_cfg.get("media_type", "unknown")
        cache_path = get_cache_dir() / "datasets" / modality / name
        marker = cache_path / ".download_complete"

        n = _count_media_files(cache_path, modality)

        if marker.exists():
            status = "OK"
        elif cache_path.exists() and n > 0:
            status = "PARTIAL"
        else:
            status = "MISSING"

        total_samples += n
        if media_type == "real":
            real_count += n
        else:
            fake_count += n

        print(f"{name:<35} {media_type:<12} {n:>8} {status}")

    print("-" * 70)
    print(f"{'Total':<35} {'':12} {total_samples:>8}")
    print(f"  Real: {real_count}  |  Synthetic/Semi: {fake_count}")
    if total_samples > 0:
        balance = min(real_count, fake_count) / max(real_count, fake_count) if max(real_count, fake_count) > 0 else 0
        print(f"  Balance ratio: {balance:.2f} (1.0 = perfect)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for dfresearch"
    )
    parser.add_argument(
        "--modality",
        choices=["image", "video", "audio", "image_human", "all"],
        default="all",
        help="Which modality to download (default: all)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify cache status instead of downloading",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent download workers (default: 4)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max samples to download per dataset (default: 500)",
    )
    parser.add_argument(
        "--refresh-configs",
        action="store_true",
        help="Force re-fetch dataset configs from gasbench GitHub",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Directory for datasets and config cache (default: ~/.cache/dfresearch or $DFRESEARCH_CACHE)",
    )
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["DFRESEARCH_CACHE"] = str(Path(args.cache_dir).expanduser())

    if args.refresh_configs:
        from dfresearch.data import refresh_configs
        refresh_configs()
        return

    modalities = ["image", "video", "audio", "image_human"] if args.modality == "all" else [args.modality]

    for mod in modalities:
        if args.verify:
            verify_cache(mod)
        else:
            download_datasets(mod, max_workers=args.workers, max_samples=args.max_samples)


if __name__ == "__main__":
    main()

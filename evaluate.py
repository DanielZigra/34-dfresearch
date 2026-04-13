#!/usr/bin/env python3
"""
evaluate.py — Run gasbench-compatible evaluation on trained models.

Evaluates a checkpoint against the local validation set using the same
sn34_score metric that BitMind Subnet 34 uses for scoring.

Usage:
    uv run evaluate.py --modality image
    uv run evaluate.py --modality image --model clip-vit-l14
    uv run evaluate.py --modality video --weights path/to/model.safetensors
    uv run evaluate.py --modality audio --model ast
"""

import argparse
import os
import time
from pathlib import Path

import torch

from prepare import (
    TARGET_IMAGE_SIZE,
    TARGET_VIDEO_SIZE,
    NUM_VIDEO_FRAMES,
    evaluate_model,
    compute_sn34_score,
)


def load_model_from_checkpoint(
    modality: str,
    model_name: str,
    weights_path: Path,
    device: str = "cuda",
):
    """Load a model from a safetensors checkpoint."""
    from dfresearch.models import get_model
    from safetensors.torch import load_file

    model = get_model(modality, model_name, num_classes=2, pretrained=False)
    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument("--modality", required=True, choices=["image", "video", "audio"])
    parser.add_argument("--model", default=None, help="Model name (default: auto-detect from modality)")
    parser.add_argument("--weights", type=Path, default=None, help="Path to model.safetensors")
    parser.add_argument("--batch-size", type=int, default=None)
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

    DEFAULTS = {
        "image": {"model": "efficientnet-b4", "batch_size": 64},
        "video": {"model": "r3d-18", "batch_size": 4},
        "audio": {"model": "wav2vec2", "batch_size": 32},
    }

    model_name = args.model or DEFAULTS[args.modality]["model"]
    batch_size = args.batch_size or DEFAULTS[args.modality]["batch_size"]
    weights_path = args.weights or Path(f"results/checkpoints/{args.modality}_{args.model}/model.safetensors")

    if not weights_path.exists():
        print(f"ERROR: Weights not found at {weights_path}")
        print(f"Run train_{args.modality}.py first, or specify --weights")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Weights: {weights_path}")

    # Load model
    print("\nLoading model...")
    model = load_model_from_checkpoint(args.modality, model_name, weights_path, device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params / 1e6:.1f}M")

    # Load validation data
    from dfresearch.data import make_dataloader

    print("Loading validation data...")
    loader_kwargs = {
        "batch_size": batch_size,
        "augment_level": 0,
    }
    if args.modality in ("image", "video"):
        target_size = TARGET_IMAGE_SIZE if args.modality == "image" else TARGET_VIDEO_SIZE
        loader_kwargs["target_size"] = target_size
    if args.modality == "video":
        loader_kwargs["num_frames"] = NUM_VIDEO_FRAMES

    val_loader = make_dataloader(args.modality, split="val", **loader_kwargs)
    print(f"Validation batches: {len(val_loader)}")

    # Evaluate
    print("\nEvaluating...")
    t0 = time.time()
    metrics = evaluate_model(model, val_loader, device=device)
    eval_time = time.time() - t0

    # Results
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results — {args.modality} / {model_name}")
    print(f"{'=' * 60}")
    print(f"sn34_score:  {metrics['sn34_score']:.6f}")
    print(f"accuracy:    {metrics['accuracy']:.6f}")
    print(f"mcc:         {metrics['mcc']:.6f}")
    print(f"brier:       {metrics['brier']:.6f}")
    print(f"mcc_norm:    {metrics['mcc_norm']:.6f}")
    print(f"brier_norm:  {metrics['brier_norm']:.6f}")
    print(f"eval_time:   {eval_time:.1f}s")
    print(f"{'=' * 60}")

    # Competition readiness check
    passing = metrics["accuracy"] >= 0.80
    print(f"\nEntrance exam threshold (>=80% accuracy): {'PASS' if passing else 'FAIL'}")
    if not passing:
        print("  Your model needs at least 80% accuracy to pass the gasbench entrance exam.")
        print("  Keep training or try a different approach.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train_image_human.py — Human-face real vs synthetic image training (Hugging Face backbones).

Same training loop as train_image.py, but modality `image_human` (datasets under
datasets/image_human.yaml, plus gasbench `image_human_datasets.yaml`) and Hub checkpoints.

Available models:
  deepfake-detector-v2      — prithivMLmods/Deep-Fake-Detector-v2-Model (ViT-B/16)
  airealnet                 — Modotte/AIRealNet (Swinv2; 224 training vs 256 pretrain, interpolate_pos_encoding)
  real-vs-ai-face           — ruhul590/real-vs-ai-human-face-classify
  human-faces-ai-vs-real    — dima806/human_faces_ai_vs_real_image_detection
  siglip-dinov2-ensemble    — Bombek1/ai-image-detector-siglip-dinov2 (SigLIP2 + DINOv2; 384/392 branches)

Usage:
    uv run train_image_human.py
    uv run train_image_human.py --model airealnet --epochs 15
    uv run train_image_human.py --model real-vs-ai-face --lr-scheduler none --early-stopping-patience 0
    uv run train_image_human.py --time-budget 600   # optional wall-clock cap (with --epochs)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import torch
import torch.nn.functional as F
import yaml

try:
    import wandb
    WANDB_AVAILABLE = wandb.api.api_key is not None
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

from prepare import (
    TARGET_IMAGE_SIZE,
    DEFAULT_IMAGE_BATCH_SIZE,
    evaluate_model,
)

# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS — The agent tunes these
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "deepfake-detector-v2"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = DEFAULT_IMAGE_BATCH_SIZE
AUGMENT_LEVEL = 2
MAX_PER_CLASS = None
WARMUP_STEPS = 0
GRAD_ACCUM_STEPS = 1
DEFAULT_EPOCHS = 20
USE_AMP = True
FREEZE_BACKBONE = False
DROPOUT = 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of full passes over the training set (primary stop criterion).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on optimizer steps (stops training early when reached).",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=None,
        help="Optional wall-clock training cap in seconds (stops when elapsed time exceeds).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=WARMUP_STEPS,
        help="Linear LR warmup from 0 to --lr over this many steps (0 = off).",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "plateau"),
        default="plateau",
        help="plateau: ReduceLROnPlateau on val sn34_score; none: constant LR until epoch end.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=2,
        help="Plateau scheduler: epochs with no val sn34 improvement before reducing LR.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="Plateau scheduler: multiply LR by this factor when reducing.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop if val sn34_score does not improve for this many epochs (0 = disabled).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for datasets and config cache (default: ~/.cache/dfresearch)",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    if args.cache_dir:
        from dfresearch.data import set_cache_dir

        set_cache_dir(args.cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.reset_peak_memory_stats()

    from dfresearch.models import get_model

    model = get_model("image_human", args.model, num_classes=2, pretrained=True, dropout=DROPOUT)

    if FREEZE_BACKBONE:
        inner = getattr(model, "model", None)
        if inner is not None:
            for param in inner.parameters():
                param.requires_grad = False

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} ({num_params / 1e6:.1f}M params, {num_trainable / 1e6:.1f}M trainable)")

    from dfresearch.data import make_dataloader

    train_loader = make_dataloader(
        "image_human",
        split="train",
        batch_size=args.batch_size,
        target_size=TARGET_IMAGE_SIZE,
        augment_level=AUGMENT_LEVEL,
        max_per_class=MAX_PER_CLASS,
    )
    val_loader = make_dataloader(
        "image_human",
        split="val",
        batch_size=args.batch_size * 2,
        target_size=TARGET_IMAGE_SIZE,
        augment_level=0,
        max_per_class=MAX_PER_CLASS,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}", flush=True)

    if len(train_loader) == 0:
        print(
            "ERROR: No training data. Copy datasets/image_human.yaml.example to datasets/image_human.yaml, "
            "add datasets, then run `uv run prepare.py --modality image_human`."
        )
        sys.exit(1)

    print("Setting up optimizer, AMP, and (if enabled) W&B — this can take a few seconds...", flush=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    lr_scheduler = None
    if args.lr_scheduler == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=1e-8,
        )

    amp_enabled = USE_AMP and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if WANDB_AVAILABLE:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "dfresearch"),
            config={
                "modality": "image_human",
                "model": args.model,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "time_budget": args.time_budget,
                "augment_level": AUGMENT_LEVEL,
                "warmup_steps": args.warmup_steps,
                "grad_accum": GRAD_ACCUM_STEPS,
                "dropout": DROPOUT,
                "freeze_backbone": FREEZE_BACKBONE,
                "weight_decay": WEIGHT_DECAY,
                "max_per_class": MAX_PER_CLASS,
                "lr_scheduler": args.lr_scheduler,
                "scheduler_patience": args.scheduler_patience,
                "scheduler_factor": args.scheduler_factor,
                "early_stopping_patience": args.early_stopping_patience,
                "num_params_M": round(num_params / 1e6, 1),
                "num_trainable_M": round(num_trainable / 1e6, 1),
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
            },
            tags=["image_human", args.model],
            reinit=True,
        )
        print(f"W&B: logging to {wandb.run.url}", flush=True)

    from tqdm import tqdm

    step = 0
    epoch = 0
    t_start = time.time()
    is_tty = sys.stdout.isatty()

    stop_reason = None
    best_sn34 = float("-inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    caps = []
    if args.epochs:
        caps.append(f"{args.epochs} epochs")
    if args.max_steps is not None:
        caps.append(f"{args.max_steps} steps max")
    if args.time_budget is not None:
        caps.append(f"{args.time_budget}s wall time")
    print(f"\nTraining ({', '.join(caps)})...", flush=True)
    print(
        "Tip: the first training step can take tens of seconds on a large backbone before tqdm moves.",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        if stop_reason:
            break

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_tty, leave=False, ncols=100)
        for batch_inputs, batch_labels in pbar:
            elapsed = time.time() - t_start
            if args.time_budget is not None and elapsed >= args.time_budget:
                stop_reason = "time_budget"
                break
            if args.max_steps is not None and step >= args.max_steps:
                stop_reason = "max_steps"
                break

            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            if args.warmup_steps > 0 and step < args.warmup_steps:
                lr_scale = (step + 1) / args.warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * lr_scale

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(batch_inputs)
                loss = F.cross_entropy(logits, batch_labels)
                loss = loss / GRAD_ACCUM_STEPS

            if torch.isnan(loss):
                print("ERROR: NaN loss detected, aborting.")
                sys.exit(1)

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_loss = loss.item() * GRAD_ACCUM_STEPS
            epoch_loss += batch_loss
            epoch_steps += 1
            step += 1

            lr = optimizer.param_groups[0]["lr"]
            postfix = f"loss={epoch_loss / epoch_steps:.4f} lr={lr:.1e}"
            if args.time_budget is not None:
                postfix += f" rem={max(0, args.time_budget - elapsed):.0f}s"
            pbar.set_postfix_str(postfix)
        pbar.close()

        if epoch_steps == 0:
            break

        elapsed = time.time() - t_start
        avg_loss = epoch_loss / epoch_steps
        lr = optimizer.param_groups[0]["lr"]

        val_metrics = evaluate_model(model, val_loader, device=device)
        if lr_scheduler is not None:
            lr_scheduler.step(val_metrics["sn34_score"])

        improved = val_metrics["sn34_score"] > best_sn34
        if improved:
            best_sn34 = val_metrics["sn34_score"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:<4d}/{args.epochs} | train_loss={avg_loss:.4f} | "
            f"val_sn34={val_metrics['sn34_score']:.4f} | lr={lr:.1e} | step={step} | {elapsed:.0f}s",
            flush=True,
        )
        if WANDB_AVAILABLE:
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/epoch": epoch,
                    "train/step": step,
                    "val/sn34_score": val_metrics["sn34_score"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/mcc": val_metrics["mcc"],
                    "val/brier": val_metrics["brier"],
                }
            )

        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            stop_reason = "early_stopping"
            print(
                f"Early stopping: no val sn34 improvement for {args.early_stopping_patience} epochs "
                f"(best epoch {best_epoch}, sn34={best_sn34:.6f}).",
                flush=True,
            )
            break

        if stop_reason == "time_budget" or stop_reason == "max_steps":
            break

    training_seconds = time.time() - t_start

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"\nLoaded best checkpoint (epoch {best_epoch}, val sn34={best_sn34:.6f}).", flush=True)

    print("\nEvaluating...")
    t_eval = time.time()
    metrics = evaluate_model(model, val_loader, device=device)
    eval_seconds = time.time() - t_eval

    total_seconds = training_seconds + eval_seconds
    peak_vram_mb = 0.0
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"\n{'=' * 60}")
    print("---")
    print(f"model:            {args.model}")
    print(f"sn34_score:       {metrics['sn34_score']:.6f}")
    print(f"accuracy:         {metrics['accuracy']:.6f}")
    print(f"mcc:              {metrics['mcc']:.6f}")
    print(f"brier:            {metrics['brier']:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"num_epochs:       {epoch}  (best val: epoch {best_epoch})")
    if stop_reason:
        print(f"stop_reason:      {stop_reason}")
    print(f"batch_size:       {args.batch_size}")
    print(f"learning_rate:    {args.lr}")
    print(f"augment_level:    {AUGMENT_LEVEL}")

    if WANDB_AVAILABLE:
        wandb.log(
            {
                "eval/sn34_score": metrics["sn34_score"],
                "eval/accuracy": metrics["accuracy"],
                "eval/mcc": metrics["mcc"],
                "eval/brier": metrics["brier"],
                "system/peak_vram_mb": peak_vram_mb,
                "system/training_seconds": training_seconds,
            }
        )
        wandb.summary.update({"sn34_score": metrics["sn34_score"], "accuracy": metrics["accuracy"]})

    from safetensors.torch import save_file
    from export import generate_model_config, generate_model_py

    ckpt_dir = Path("results") / "checkpoints" / "image_human"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), ckpt_dir / "model.safetensors")

    config = generate_model_config("image_human", args.model)
    with open(ckpt_dir / "model_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    (ckpt_dir / "model.py").write_text(generate_model_py("image_human", args.model))

    print(f"\nCheckpoint saved to {ckpt_dir}/")
    print("  model.safetensors, model.py, model_config.yaml — ready for submission")

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta = {
        "timestamp": ts,
        "modality": "image_human",
        "model": args.model,
        "sn34_score": metrics["sn34_score"],
        "accuracy": metrics["accuracy"],
        "mcc": metrics["mcc"],
        "brier": metrics["brier"],
        "training_seconds": training_seconds,
        "peak_vram_mb": peak_vram_mb,
        "num_steps": step,
        "num_epochs_ran": epoch,
        "best_val_epoch": best_epoch,
        "stop_reason": stop_reason,
        "num_params_M": round(num_params / 1e6, 1),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "augment_level": AUGMENT_LEVEL,
    }
    (runs_dir / f"{ts}_meta.json").write_text(json.dumps(run_meta, indent=2))

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()

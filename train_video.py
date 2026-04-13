#!/usr/bin/env python3
"""
train_video.py — Video deepfake detection training script.

THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.
Everything is fair game: model choice, hyperparameters, optimizer,
frame sampling, temporal aggregation, architecture tweaks, etc.

The goal: maximize sn34_score on the validation set within the time budget.

Available models: r3d-18, videomae, hiera

Usage:
    uv run train_video.py
    uv run train_video.py --model videomae
    uv run train_video.py --model hiera
    uv run train_video.py > run.log 2>&1
"""

import argparse
import time
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

try:
    import wandb
    WANDB_AVAILABLE = wandb.api.api_key is not None
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

from prepare import (
    TIME_BUDGET,
    TARGET_VIDEO_SIZE,
    NUM_VIDEO_FRAMES,
    DEFAULT_VIDEO_BATCH_SIZE,
    evaluate_model,
)

# ──────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS — The agent tunes these
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "r3d-18"                  # "r3d-18" or "videomae"
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = DEFAULT_VIDEO_BATCH_SIZE  # 4
NUM_FRAMES = NUM_VIDEO_FRAMES          # 16
AUGMENT_LEVEL = 1                      # 0=none, 1=basic, 2=medium, 3=hard
MAX_PER_CLASS = 2000
WARMUP_STEPS = 50
GRAD_ACCUM_STEPS = 4                   # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
USE_AMP = True
FREEZE_BACKBONE = False
DROPOUT = 0.3

# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--time-budget", type=int, default=TIME_BUDGET)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for datasets and config cache (default: ~/.cache/dfresearch)",
    )
    args = parser.parse_args()

    if args.cache_dir:
        from dfresearch.data import set_cache_dir

        set_cache_dir(args.cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.reset_peak_memory_stats()

    # ── Model ──
    from dfresearch.models import get_model
    model = get_model("video", args.model, num_classes=2, pretrained=True, dropout=DROPOUT)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} ({num_params / 1e6:.1f}M params, {num_trainable / 1e6:.1f}M trainable)")

    # ── Data ──
    from dfresearch.data import make_dataloader

    train_loader = make_dataloader(
        "video", split="train", batch_size=args.batch_size,
        target_size=TARGET_VIDEO_SIZE, num_frames=NUM_FRAMES,
        augment_level=AUGMENT_LEVEL, max_per_class=MAX_PER_CLASS,
    )
    val_loader = make_dataloader(
        "video", split="val", batch_size=args.batch_size,
        target_size=TARGET_VIDEO_SIZE, num_frames=NUM_FRAMES,
        augment_level=0, max_per_class=MAX_PER_CLASS,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    if len(train_loader) == 0:
        print("ERROR: No training data. Run `uv run prepare.py --modality video` first.")
        sys.exit(1)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    amp_enabled = USE_AMP and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ── W&B init ──
    if WANDB_AVAILABLE:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "dfresearch"),
            config={
                "modality": "video", "model": args.model,
                "lr": args.lr, "batch_size": args.batch_size,
                "time_budget": args.time_budget, "augment_level": AUGMENT_LEVEL,
                "num_frames": NUM_FRAMES, "warmup_steps": WARMUP_STEPS,
                "grad_accum": GRAD_ACCUM_STEPS, "weight_decay": WEIGHT_DECAY,
                "num_params_M": round(num_params / 1e6, 1),
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
            },
            tags=["video", args.model],
            reinit=True,
        )
        print(f"W&B: logging to {wandb.run.url}", flush=True)

    # ── Training loop ──
    from tqdm import tqdm

    model.train()
    step = 0
    epoch = 0
    total_loss = 0.0
    t_start = time.time()
    budget = args.time_budget
    is_tty = sys.stdout.isatty()

    print(f"\nTraining for {budget}s budget...", flush=True)

    time_up = False
    while not time_up:
        epoch += 1
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_tty, leave=False, ncols=100)
        for batch_inputs, batch_labels in pbar:
            elapsed = time.time() - t_start
            if elapsed >= budget:
                time_up = True
                break

            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            if WARMUP_STEPS > 0 and step < WARMUP_STEPS:
                lr_scale = (step + 1) / WARMUP_STEPS
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
            total_loss += batch_loss
            epoch_loss += batch_loss
            epoch_steps += 1
            step += 1

            lr = optimizer.param_groups[0]["lr"]
            remaining = max(0, budget - elapsed)
            pbar.set_postfix_str(f"loss={epoch_loss / epoch_steps:.4f} lr={lr:.1e} rem={remaining:.0f}s")
        pbar.close()

        elapsed = time.time() - t_start
        if epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:<4d} | loss={avg_loss:.4f} | lr={lr:.1e} | step={step} | {elapsed:.0f}s/{budget}s", flush=True)
            if WANDB_AVAILABLE:
                wandb.log({"train/loss": avg_loss, "train/lr": lr, "train/epoch": epoch, "train/step": step})

    training_seconds = time.time() - t_start

    # ── Evaluation ──
    print("\nEvaluating...")
    t_eval = time.time()
    metrics = evaluate_model(model, val_loader, device=device)
    eval_seconds = time.time() - t_eval

    total_seconds = training_seconds + eval_seconds
    peak_vram_mb = 0.0
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # ── Output summary ──
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
    print(f"num_epochs:       {epoch}")
    print(f"num_frames:       {NUM_FRAMES}")
    print(f"batch_size:       {args.batch_size}")
    print(f"learning_rate:    {args.lr}")
    print(f"augment_level:    {AUGMENT_LEVEL}")

    if WANDB_AVAILABLE:
        wandb.log({"eval/sn34_score": metrics["sn34_score"], "eval/accuracy": metrics["accuracy"],
                    "eval/mcc": metrics["mcc"], "eval/brier": metrics["brier"],
                    "system/peak_vram_mb": peak_vram_mb, "system/training_seconds": training_seconds})
        wandb.summary.update({"sn34_score": metrics["sn34_score"], "accuracy": metrics["accuracy"]})

    from safetensors.torch import save_file
    from pathlib import Path
    import json
    from datetime import datetime

    ckpt_dir = Path("results") / "checkpoints" / "video"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), ckpt_dir / "model.safetensors")

    from export import generate_model_config, generate_model_py
    config = generate_model_config("video", args.model)
    with open(ckpt_dir / "model_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    (ckpt_dir / "model.py").write_text(generate_model_py("video", args.model))

    print(f"\nCheckpoint saved to {ckpt_dir}/")
    print(f"  model.safetensors, model.py, model_config.yaml — ready for submission")

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta = {
        "timestamp": ts,
        "modality": "video",
        "model": args.model,
        "sn34_score": metrics["sn34_score"],
        "accuracy": metrics["accuracy"],
        "mcc": metrics["mcc"],
        "brier": metrics["brier"],
        "training_seconds": training_seconds,
        "peak_vram_mb": peak_vram_mb,
        "num_steps": step,
        "num_params_M": round(num_params / 1e6, 1),
        "num_frames": NUM_FRAMES,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "augment_level": AUGMENT_LEVEL,
    }
    (runs_dir / f"{ts}_meta.json").write_text(json.dumps(run_meta, indent=2))

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train_clip_h2.py — Stage A (linear probe) training for CLIPViTDetector.

Stage A: backbone frozen, only head_binary + head_gentype are trained.
Gradients never flow back into the ViT layers.

Loss:  L = CE(binary) + 0.3 * CE(gentype) + 0.1 * SupCon(embeddings)
       Label smoothing = 0.1  (baked into DetectionLoss defaults)

Generator-type labels are derived from binary labels:
  real      → GENERATOR_CLASSES["real"]    (index 0)
  synthetic → GENERATOR_CLASSES["unknown"] (index 4)

Dataset loading follows the same pattern as train_image.py:
  set_cache_dir() → gather_samples("image_human") → DataLoader

After training:
  - Validates on the local val split using the sn34_score metric.
  - Saves submission-ready weights via save_submission_weights()
    (head_gentype excluded, head_binary + vision backbone included).

Usage:
    uv run train_clip_h2.py --cache-dir /workspace/.cache
    uv run train_clip_h2.py --cache-dir /workspace/.cache --epochs 10 --batch-size 64
    uv run train_clip_h2.py --cache-dir /workspace/.cache --mixup --augment-level 2
    uv run train_clip_h2.py  # uses ~/.cache/dfresearch (default)
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import wandb
    WANDB_AVAILABLE = wandb.api.api_key is not None
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

from prepare import TARGET_IMAGE_SIZE, DEFAULT_IMAGE_BATCH_SIZE, evaluate_model

# ── Stage A hyperparameters ───────────────────────────────────────────────────
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4
BATCH_SIZE       = DEFAULT_IMAGE_BATCH_SIZE   # 32
AUGMENT_LEVEL    = 1
WARMUP_STEPS     = 50
GRAD_ACCUM_STEPS = 1
DEFAULT_EPOCHS   = 5
USE_AMP          = True
MIXUP_ALPHA      = 0.4   # active only when --mixup is set
MAX_PER_CLASS    = None  # set to int to cap samples per class

# Map binary label → generator-type class index (GENERATOR_CLASSES order)
# GENERATOR_CLASSES = ["real", "gan", "diffusion", "faceswap", "unknown"]
GENTYPE_FROM_BINARY = {0: 0, 1: 4}   # real→0,  synthetic→"unknown"=4


# ── Dataset: wraps ImageDeepfakeDataset to also yield generator-type labels ───

class StageADataset(torch.utils.data.Dataset):
    """
    Thin wrapper around the standard image cache that adds a gentype label.

    The base dataset is built from gather_samples() output — the same source
    used by make_dataloader() in train_image.py.

    Returns (image_tensor [uint8 CHW], binary_label, gentype_label).
      binary_label  : 0=real, 1=synthetic
      gentype_label : 0=real, 4=unknown  (derived from binary_label)
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        target_size: tuple[int, int] = (224, 224),
        augment_level: int = 0,
    ):
        from dfresearch.data import ImageDeepfakeDataset
        self._base = ImageDeepfakeDataset(
            samples, target_size=target_size, augment_level=augment_level,
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        tensor, binary_label = self._base[idx]
        gentype_label = GENTYPE_FROM_BINARY[int(binary_label)]
        return tensor, binary_label, gentype_label


def _make_train_loader(
    samples: list[tuple[Path, int]],
    batch_size: int,
    augment_level: int,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    ds = StageADataset(samples, target_size=TARGET_IMAGE_SIZE, augment_level=augment_level)
    eff_workers = min(num_workers, len(ds))
    return torch.utils.data.DataLoader(
        ds,
        batch_size=min(batch_size, max(len(ds), 1)),
        shuffle=True,
        num_workers=eff_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(ds) > batch_size,
        persistent_workers=(eff_workers > 0),
    )


# ── Mixup helper ──────────────────────────────────────────────────────────────

def mixup_batch(
    x: torch.Tensor,       # [B, 3, H, W] float32
    y_bin: torch.Tensor,   # [B]  long
    y_gen: torch.Tensor,   # [B]  long
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup augmentation to a batch.

    Returns mixed_x, y_a_bin, y_b_bin, y_a_gen, y_b_gen, lam.
    The caller computes:  loss = lam * L(pred, a) + (1-lam) * L(pred, b)
    """
    lam  = float(np.random.beta(alpha, alpha))
    idx  = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y_bin, y_bin[idx], y_gen, y_gen[idx], lam


# ── Gradient sanity check ─────────────────────────────────────────────────────

def _assert_backbone_frozen(model) -> None:
    """Raise if any backbone parameter has requires_grad=True."""
    for name, param in model.vision.named_parameters():
        if param.requires_grad:
            raise RuntimeError(
                f"Backbone parameter '{name}' is not frozen! "
                "Stage A requires freeze_backbone=True."
            )


def _count_trainable(model) -> tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage A linear probe for CLIPViTDetector (backbone frozen)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay",type=float, default=WEIGHT_DECAY)
    parser.add_argument("--augment-level", type=int, default=AUGMENT_LEVEL,
                        help="0=none 1=basic 2=medium 3=hard")
    parser.add_argument("--warmup-steps", type=int,  default=WARMUP_STEPS)
    parser.add_argument("--grad-accum",   type=int,  default=GRAD_ACCUM_STEPS)
    parser.add_argument("--mixup", action="store_true",
                        help=f"Enable Mixup augmentation (alpha={MIXUP_ALPHA})")
    parser.add_argument("--cache-dir", default=None, metavar="PATH",
                        help="Dataset cache root. When given, datasets are read "
                             "directly via dataset_info.json — no network required.")
    parser.add_argument("--output-dir", default="results/checkpoints/clip_h2",
                        help="Directory to write model.safetensors and run artifacts.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.reset_peak_memory_stats()

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding CLIPViTDetector (pretrained=True, freeze_backbone=True)...")
    from dfresearch.models.image_human.clip_h2 import (
        CLIPViTDetector, DetectionLoss, save_submission_weights,
        NUM_GENERATOR_CLASSES,
    )

    model = CLIPViTDetector(pretrained=True, freeze_backbone=True)
    _assert_backbone_frozen(model)   # hard-fail if any ViT param is trainable
    model = model.to(device)

    total_params, trainable_params = _count_trainable(model)
    print(f"Params : {total_params / 1e6:.1f}M total, "
          f"{trainable_params / 1e6:.2f}M trainable  "
          f"({100 * trainable_params / total_params:.1f}%)")
    print(f"Trainable modules: head_binary, head_gentype")
    print(f"Frozen   modules : vision (ViT-B/16 backbone — {(total_params - trainable_params) / 1e6:.1f}M params)")

    # ── Data  (same pattern as train_image.py) ───────────────────────────────
    if args.cache_dir:
        from dfresearch.data import set_cache_dir
        set_cache_dir(args.cache_dir)

    from dfresearch.data import gather_samples, make_dataloader

    # gather_samples gives us (Path, label) pairs — identical to train_image.py
    train_samples = gather_samples(
        "image_human", split="train", max_per_class=MAX_PER_CLASS, seed=args.seed,
    )
    val_samples = gather_samples(
        "image_human", split="val", max_per_class=MAX_PER_CLASS, seed=args.seed,
    )

    if not train_samples:
        print(
            "ERROR: No cached image_human samples found.\n"
            "  Run:  uv run prepare.py --modality image_human\n"
            "  or pass --cache-dir pointing at a prepared dataset root.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_real_train = sum(1 for _, l in train_samples if l == 0)
    n_fake_train = sum(1 for _, l in train_samples if l == 1)
    n_real_val   = sum(1 for _, l in val_samples   if l == 0)
    n_fake_val   = sum(1 for _, l in val_samples   if l == 1)

    print(f"Train  : {len(train_samples):,}  (real={n_real_train}, synthetic={n_fake_train})")
    print(f"Val    : {len(val_samples):,}  (real={n_real_val}, synthetic={n_fake_val})")

    # Training loader — StageADataset adds gentype labels on top of ImageDeepfakeDataset
    train_loader = _make_train_loader(
        train_samples, args.batch_size, augment_level=args.augment_level,
    )

    # Validation loader — plain make_dataloader (same as train_image.py) returning (img, label)
    val_loader = make_dataloader(
        "image_human", split="val",
        batch_size=args.batch_size * 2,
        target_size=TARGET_IMAGE_SIZE,
        augment_level=0,
        max_per_class=MAX_PER_CLASS,
    )

    print(f"Batches: {len(train_loader)} train / {len(val_loader)} val  "
          f"(batch_size={args.batch_size})")

    # ── Loss & optimizer ──────────────────────────────────────────────────────
    criterion = DetectionLoss(lambda_gen=0.3, lambda_con=0.1, label_smoothing=0.1)

    # Only head_binary + head_gentype parameters are trainable
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params_list,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    amp_enabled = USE_AMP and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ── W&B ───────────────────────────────────────────────────────────────────
    if WANDB_AVAILABLE:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "dfresearch"),
            config={
                "stage": "A_linear_probe",
                "model": "clip-vit-b16-dual-head",
                "lr": args.lr, "weight_decay": args.weight_decay,
                "batch_size": args.batch_size, "epochs": args.epochs,
                "augment_level": args.augment_level,
                "warmup_steps": args.warmup_steps,
                "grad_accum": args.grad_accum,
                "mixup": args.mixup,
                "freeze_backbone": True,
                "total_params_M": round(total_params / 1e6, 1),
                "trainable_params_M": round(trainable_params / 1e6, 2),
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
            },
            tags=["image_human", "clip-h2", "stage-A"],
            reinit=True,
        )
        print(f"W&B: {wandb.run.url}")

    # ── Training loop ─────────────────────────────────────────────────────────
    from tqdm import tqdm

    best_sn34   = float("-inf")
    best_state  = None
    best_epoch  = 0
    step        = 0
    t_start     = time.time()
    is_tty      = sys.stdout.isatty()

    print(f"\nStage A — linear probe  ({args.epochs} epochs, lr={args.lr:.0e})")
    print("Backbone : FROZEN  (gradients stop at head input)\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = epoch_bin = epoch_gen = epoch_con = 0.0
        epoch_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{args.epochs}",
            disable=not is_tty,
            leave=False,
            ncols=110,
        )

        for imgs_uint8, labels_bin, labels_gen in pbar:
            imgs_uint8  = imgs_uint8.to(device, non_blocking=True)
            labels_bin  = labels_bin.to(device, non_blocking=True)
            labels_gen  = labels_gen.to(device, non_blocking=True)

            # Cast to float32 so forward_train outputs float32 (not uint8)
            imgs_f32 = imgs_uint8.float()

            # Mixup (optional)
            if args.mixup:
                imgs_f32, y_bin_a, y_bin_b, y_gen_a, y_gen_b, lam = mixup_batch(
                    imgs_f32, labels_bin, labels_gen, alpha=MIXUP_ALPHA,
                )

            # Warmup LR
            if args.warmup_steps > 0 and step < args.warmup_steps:
                lr_scale = (step + 1) / args.warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * lr_scale

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits_bin, logits_gen, feat_norm = model.forward_train(imgs_f32)

                if args.mixup:
                    loss_a, info_a = criterion(logits_bin, logits_gen, feat_norm,
                                               y_bin_a, y_gen_a)
                    loss_b, _      = criterion(logits_bin, logits_gen, feat_norm,
                                               y_bin_b, y_gen_b)
                    loss  = lam * loss_a + (1 - lam) * loss_b
                    info  = info_a   # log the "a" breakdown for simplicity
                else:
                    loss, info = criterion(logits_bin, logits_gen, feat_norm,
                                          labels_bin, labels_gen)

                loss = loss / args.grad_accum

            if torch.isnan(loss):
                print("ERROR: NaN loss — aborting.", file=sys.stderr)
                sys.exit(1)

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_loss  = loss.item() * args.grad_accum
            epoch_loss += batch_loss
            epoch_bin  += info["loss_binary"]
            epoch_gen  += info["loss_gentype"]
            epoch_con  += info["loss_supcon"]
            epoch_steps += 1
            step += 1

            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix_str(
                f"loss={epoch_loss/epoch_steps:.3f}  "
                f"bin={epoch_bin/epoch_steps:.3f}  "
                f"gen={epoch_gen/epoch_steps:.3f}  "
                f"con={epoch_con/epoch_steps:.3f}  "
                f"lr={lr:.1e}"
            )
        pbar.close()

        if epoch_steps == 0:
            break

        # ── Epoch summary ──────────────────────────────────────────────────
        elapsed = time.time() - t_start
        avg_loss = epoch_loss / epoch_steps
        lr = optimizer.param_groups[0]["lr"]

        # Evaluate on val set (uses model.forward() → binary logits only)
        val_metrics = _validate(model, val_loader, device)
        sn34 = val_metrics["sn34_score"]

        improved = sn34 > best_sn34
        if improved:
            best_sn34  = sn34
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  ★ best"
        else:
            marker = ""

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={avg_loss:.4f}  "
            f"val_sn34={sn34:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
            f"lr={lr:.1e}  {elapsed:.0f}s"
            f"{marker}",
            flush=True,
        )

        if WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch, "step": step,
                "train/loss_total":   avg_loss,
                "train/loss_binary":  epoch_bin  / epoch_steps,
                "train/loss_gentype": epoch_gen  / epoch_steps,
                "train/loss_supcon":  epoch_con  / epoch_steps,
                "train/lr": lr,
                "val/sn34_score": sn34,
                "val/accuracy":   val_metrics["accuracy"],
                "val/mcc":        val_metrics["mcc"],
                "val/brier":      val_metrics["brier"],
            })

    training_seconds = time.time() - t_start

    # ── Restore best checkpoint ───────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"\nRestored best checkpoint (epoch {best_epoch}, val sn34={best_sn34:.6f})")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\nFinal evaluation on val set...")
    final_metrics = _validate(model, val_loader, device)

    peak_vram_mb = 0.0
    if device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"\n{'=' * 60}")
    print(f"Stage A — CLIPViTDetector  (linear probe, backbone frozen)")
    print(f"{'=' * 60}")
    print(f"sn34_score      : {final_metrics['sn34_score']:.6f}")
    print(f"accuracy        : {final_metrics['accuracy']:.6f}")
    print(f"mcc             : {final_metrics['mcc']:.6f}")
    print(f"brier           : {final_metrics['brier']:.6f}")
    print(f"best_val_epoch  : {best_epoch}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"peak_vram_mb    : {peak_vram_mb:.1f}")
    print(f"total_steps     : {step}")
    print(f"trainable_M     : {trainable_params / 1e6:.2f}")
    print(f"{'=' * 60}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Submission weights (head_gentype excluded)
    save_submission_weights(model, str(out_dir / "model.safetensors"))

    # Full checkpoint (includes head_gentype — useful for Stage B continuation)
    from safetensors.torch import save_file as _save_file
    _save_file(model.state_dict(), str(out_dir / "model_full.safetensors"))
    print(f"Full checkpoint : {out_dir / 'model_full.safetensors'}  "
          f"(use as --weights for Stage B)")

    # Run metadata
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "timestamp": ts,
        "stage": "A_linear_probe",
        "model": "clip-vit-b16-dual-head",
        "freeze_backbone": True,
        "sn34_score": final_metrics["sn34_score"],
        "accuracy":   final_metrics["accuracy"],
        "mcc":        final_metrics["mcc"],
        "brier":      final_metrics["brier"],
        "best_val_epoch": best_epoch,
        "training_seconds": training_seconds,
        "peak_vram_mb": peak_vram_mb,
        "total_steps": step,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "augment_level": args.augment_level,
        "mixup": args.mixup,
        "warmup_steps": args.warmup_steps,
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "total_params_M": round(total_params / 1e6, 1),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    (runs_dir / f"{ts}_clip_h2_stageA.json").write_text(json.dumps(meta, indent=2))
    print(f"Run metadata    : runs/{ts}_clip_h2_stageA.json")

    if WANDB_AVAILABLE:
        wandb.log({
            "eval/sn34_score": final_metrics["sn34_score"],
            "eval/accuracy":   final_metrics["accuracy"],
            "eval/mcc":        final_metrics["mcc"],
            "eval/brier":      final_metrics["brier"],
            "system/peak_vram_mb": peak_vram_mb,
            "system/training_seconds": training_seconds,
        })
        wandb.summary.update({
            "sn34_score": final_metrics["sn34_score"],
            "accuracy":   final_metrics["accuracy"],
        })
        wandb.finish()


# ── Validation helper ─────────────────────────────────────────────────────────

def _validate(model, val_loader, device: str) -> dict:
    """
    Run inference-only forward (model.forward) and return sn34 metrics.

    val_loader yields (imgs_uint8, binary_label) — same 2-tuple as make_dataloader()
    returns in train_image.py.  We cast to float32 before passing to the model.
    """
    from prepare import compute_sn34_score

    model.eval()
    all_labels: list[int]   = []
    all_probs:  list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            imgs_uint8, labels_bin = batch[0], batch[1]   # works for 2- or 3-tuple
            imgs_f32 = imgs_uint8.float().to(device, non_blocking=True)
            logits   = model(imgs_f32)                             # [B, 2]
            probs    = torch.softmax(logits.float(), dim=-1)[:, 1] # P(synthetic)
            all_labels.extend(labels_bin.tolist())
            all_probs.extend(probs.cpu().tolist())

    model.train()
    return compute_sn34_score(np.array(all_labels), np.array(all_probs))


if __name__ == "__main__":
    main()

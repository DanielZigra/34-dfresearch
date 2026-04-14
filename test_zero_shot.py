#!/usr/bin/env python3
"""
test_zero_shot.py — Zero-shot AUC probe for CLIPViTDetector on gasbench datasets.

Runs zero_shot_auc() (Phase 0) on a sample batch drawn from the locally cached
image_human (or image) dataset.  No training required — uses the CLIP text
encoder as the classifier and reports AUC-ROC plus the Phase-0 decision:

    AUC >= 0.70  ->  features aligned, safe to skip Stage A
    AUC <  0.70  ->  run full 4-stage curriculum

When --cache-dir is given the script scans that directory directly via the
dataset_info.json sidecar files — no network access required.

Usage:
    uv run test_zero_shot.py
    uv run test_zero_shot.py --n-samples 300 --cache-dir /workspace/.cache
    uv run test_zero_shot.py --n-samples 200 --device cpu --cache-dir /workspace/.cache
    uv run test_zero_shot.py --real-prompt "a real photograph" \
        --fake-prompt "an AI-generated image" --cache-dir /workspace/.cache
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Decision threshold (matches zero_shot_auc docstring) ─────────────────────
AUC_SKIP_THRESHOLD = 0.70

DEFAULT_REAL_PROMPT = "a photo of a real picture"
DEFAULT_FAKE_PROMPT = "a photo of an AI-generated"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABEL_MAP = {"real": 0, "synthetic": 1, "semisynthetic": 1}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _gather_from_cache_dir(cache_dir: str | Path) -> list[tuple[str, int]]:
    """
    Scan {cache_dir}/datasets/*/  directly, using dataset_info.json for labels.

    This avoids any network requests or gasbench config fetching — everything
    is derived from the sidecar JSON written at download time.
    """
    root = Path(cache_dir) / "datasets"
    if not root.exists():
        return []

    samples: list[tuple[str, int]] = []
    for ds_dir in sorted(root.iterdir()):
        info_file = ds_dir / "dataset_info.json"
        samples_dir = ds_dir / "samples"
        if not info_file.exists() or not samples_dir.exists():
            continue

        try:
            info = json.loads(info_file.read_text())
        except Exception:
            continue

        media_type = info.get("media_type", "synthetic")
        label = LABEL_MAP.get(media_type, 1)

        for f in sorted(samples_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS and not f.name.startswith("."):
                samples.append((str(f), label))

    return samples


def _sample_balanced(
    samples: list,
    n: int,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Draw up to n balanced real/synthetic samples. Returns (paths, labels)."""
    rng = random.Random(seed)
    real = [(p, lbl) for p, lbl in samples if lbl == 0]
    fake = [(p, lbl) for p, lbl in samples if lbl == 1]

    per_class = n // 2
    real = rng.sample(real, min(per_class, len(real)))
    fake = rng.sample(fake, min(per_class, len(fake)))

    combined = real + fake
    rng.shuffle(combined)
    return [x[0] for x in combined], [x[1] for x in combined]


def _print_class_breakdown(labels: list[int]) -> None:
    print(f"  real      : {labels.count(0)}")
    print(f"  synthetic : {labels.count(1)}")
    print(f"  total     : {len(labels)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot AUC probe for CLIPViTDetector (Phase 0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Total samples to probe (balanced real/synthetic).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--real-prompt",
        default=DEFAULT_REAL_PROMPT,
        help="CLIP text prompt for the real class.",
    )
    parser.add_argument(
        "--fake-prompt",
        default=DEFAULT_FAKE_PROMPT,
        help="CLIP text prompt for the synthetic class.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        metavar="PATH",
        help=(
            "Dataset cache directory. When given, datasets are read directly "
            "via dataset_info.json — no network required. "
            "(default: ~/.cache/dfresearch or $DFRESEARCH_CACHE)"
        ),
    )
    parser.add_argument(
        "--modality",
        choices=["image", "image_human"],
        default="image_human",
        help="Modality used when falling back to gather_samples() (no --cache-dir).",
    )
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # ── Gather samples ────────────────────────────────────────────────────────
    if args.cache_dir:
        # Fast path: scan cache dir directly via dataset_info.json sidecar files.
        # No gasbench config fetch, no network required.
        print(f"\nScanning cache dir: {args.cache_dir}")
        all_samples = _gather_from_cache_dir(args.cache_dir)
        source_desc = args.cache_dir
    else:
        # Fallback: use dfresearch data pipeline (requires gasbench config).
        print(f"\nGathering '{args.modality}' samples via dfresearch data pipeline...")
        from dfresearch.data import gather_samples
        all_samples = gather_samples(args.modality, split="all", seed=args.seed)
        source_desc = f"modality={args.modality}"

    if len(all_samples) == 0:
        print(
            "\nERROR: No cached image samples found.\n"
            "  - With --cache-dir: make sure the directory contains "
            "datasets/*/dataset_info.json and datasets/*/samples/ folders.\n"
            "  - Without --cache-dir: run  uv run prepare.py --modality image_human  first.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_real_total = sum(1 for _, lbl in all_samples if lbl == 0)
    n_fake_total = sum(1 for _, lbl in all_samples if lbl == 1)
    print(f"Found {len(all_samples)} images total  "
          f"(real={n_real_total}, synthetic={n_fake_total})  [{source_desc}]")

    image_paths, labels = _sample_balanced(all_samples, n=args.n_samples, seed=args.seed)

    if len(image_paths) == 0:
        print("ERROR: No samples remained after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSample breakdown (n={len(image_paths)}):")
    _print_class_breakdown(labels)

    # ── Zero-shot probe ───────────────────────────────────────────────────────
    print(f"\nPrompts:")
    print(f"  real      : \"{args.real_prompt}\"")
    print(f"  synthetic : \"{args.fake_prompt}\"")

    print(f"\nRunning zero-shot AUC probe on {len(image_paths)} images...")

    # Load CLIP once; reuse for both AUC computation and per-class breakdown.
    from transformers import CLIPModel, CLIPProcessor
    import torch.nn.functional as F
    from PIL import Image

    model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    # model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # processor  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model_clip.eval()

    txt_inputs = processor(
        text=[args.real_prompt, args.fake_prompt],
        return_tensors="pt",
        padding=True,
    )
    # Keep only keys the text_model accepts (handles API differences across
    # transformers versions).
    text_model_keys = {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
    txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()
                  if k in text_model_keys}
    with torch.no_grad():
        text_out  = model_clip.text_model(**txt_inputs)
        text_vecs = model_clip.text_projection(text_out.pooler_output)  # [2, 512]
        text_vecs = F.normalize(text_vecs, dim=-1)

    scores_all  = []
    scores_real = []
    scores_fake = []

    t0 = time.time()
    for path, label in zip(image_paths, labels):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        inp = processor(images=img, return_tensors="pt")
        vision_keys = {"pixel_values"}
        inp = {k: v.to(device) for k, v in inp.items() if k in vision_keys}
        with torch.no_grad():
            vision_out = model_clip.vision_model(**inp)
            img_vec    = model_clip.visual_projection(vision_out.pooler_output)
            img_vec    = F.normalize(img_vec, dim=-1)
        sim  = (img_vec @ text_vecs.T).squeeze(0)
        prob = sim.softmax(dim=-1)[1].item()   # P(synthetic)
        scores_all.append(prob)
        if label == 0:
            scores_real.append(prob)
        else:
            scores_fake.append(prob)
    elapsed = time.time() - t0

    # Compute AUC from collected scores (same formula as zero_shot_auc).
    pairs  = sorted(zip(scores_all, labels), reverse=True)
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        tp = fp = auc = prev_fp = prev_tp = 0.0
        for _, lbl in pairs:
            if lbl == 1:
                tp += 1
            else:
                fp += 1
            auc += (tp + prev_tp) / 2 * (fp - prev_fp)
            prev_tp, prev_fp = tp, fp
        auc /= n_pos * n_neg

    def _stats(vals: list[float]) -> str:
        if not vals:
            return "n/a"
        import statistics
        return (
            f"mean={sum(vals)/len(vals):.3f}  "
            f"med={statistics.median(vals):.3f}  "
            f"min={min(vals):.3f}  max={max(vals):.3f}"
        )

    # ── Results ───────────────────────────────────────────────────────────────
    decision = "SKIP Stage A  (features aligned)" if auc >= AUC_SKIP_THRESHOLD \
               else "RUN full 4-stage curriculum"

    print(f"\n{'=' * 60}")
    print(f"Zero-shot AUC probe — CLIPViTDetector (Phase 0)")
    print(f"{'=' * 60}")
    print(f"source        : {source_desc}")
    print(f"n_samples     : {len(image_paths)}  (real={len(scores_real)}, synthetic={len(scores_fake)})")
    print(f"device        : {device}")
    print(f"elapsed       : {elapsed:.1f}s  ({elapsed / max(len(image_paths), 1) * 1000:.0f} ms/image)")
    print(f"")
    print(f"AUC-ROC       : {auc:.4f}  (threshold={AUC_SKIP_THRESHOLD})")
    print(f"Decision      : {decision}")
    print(f"")
    print(f"P(synthetic) for REAL images      : {_stats(scores_real)}")
    print(f"P(synthetic) for SYNTHETIC images : {_stats(scores_fake)}")
    print(f"{'=' * 60}")

    # Quick sanity check pass/fail
    if auc >= AUC_SKIP_THRESHOLD:
        print(f"\nPASS  AUC {auc:.4f} >= {AUC_SKIP_THRESHOLD}")
        sys.exit(0)
    else:
        print(f"\nINFO  AUC {auc:.4f} < {AUC_SKIP_THRESHOLD}  — full curriculum recommended")
        sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train_full.py — Full production training run.

Use this AFTER the autoresearch loop has found a good configuration.
This script:
  1. Downloads more training data (configurable samples per dataset)
  2. Runs the training script with an extended time budget
  3. Evaluates the result
  4. Exports the model for competition submission

Usage:
    # Train image model for 2 hours with 2000 samples/dataset
    uv run train_full.py --modality image --hours 2

    # Train with specific model and more data
    uv run train_full.py --modality image --model clip-vit-l14 --hours 4 --max-samples 5000

    # Train video model overnight
    uv run train_full.py --modality video --hours 8 --max-samples 3000

    # Just download more data without training
    uv run train_full.py --modality image --download-only --max-samples 5000

    # Skip download (data already cached), just train longer
    uv run train_full.py --modality audio --hours 3 --skip-download
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os


def run_cmd(
    cmd: list[str],
    desc: str,
    check: bool = True,
    cache_dir: str | None = None,
) -> subprocess.CompletedProcess:
    """Run a command with a description."""
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    env = None
    if cache_dir:
        env = os.environ.copy()
        env["DFRESEARCH_CACHE"] = str(Path(cache_dir).expanduser())
    return subprocess.run(cmd, check=check, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Full production training run after autoresearch exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick full run (1 hour, default data)
  uv run train_full.py --modality image --hours 1

  # Overnight image training
  uv run train_full.py --modality image --hours 8 --max-samples 5000

  # Multi-GPU video training (set CUDA_VISIBLE_DEVICES in .env)
  uv run train_full.py --modality video --hours 4 --max-samples 3000

  # Download more data first, then train
  uv run train_full.py --modality audio --hours 2 --max-samples 2000
""",
    )
    parser.add_argument(
        "--modality", required=True, choices=["image", "video", "audio", "image_human"],
        help="Which modality to train",
    )
    parser.add_argument(
        "--hours", type=float, default=2.0,
        help="Training time budget in hours (default: 2)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name override (default: uses whatever is set in train_{modality}.py)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=2000,
        help="Max samples per dataset to download (default: 2000, vs 500 for exploration)",
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Concurrent download workers (default: 6)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate override",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size override",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip data download step (use existing cache)",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download data, don't train",
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Skip the export step after training",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Log file path (default: runs/full_{modality}_{timestamp}.log)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Directory for datasets and config cache (default: ~/.cache/dfresearch)",
    )
    args = parser.parse_args()

    time_budget_seconds = int(args.hours * 3600)
    t_start = time.time()

    print(f"{'#' * 60}")
    print(f"#  dfresearch — Full Production Training")
    print(f"#  Modality: {args.modality}")
    print(f"#  Time budget: {args.hours}h ({time_budget_seconds}s)")
    print(f"#  Max samples/dataset: {args.max_samples}")
    if args.model:
        print(f"#  Model: {args.model}")
    if args.cache_dir:
        print(f"#  Cache: {args.cache_dir}")
    print(f"{'#' * 60}")

    # ── Step 1: Download more data ──
    if not args.skip_download:
        dl_cmd = [
            sys.executable, "prepare.py",
            "--modality", args.modality,
            "--max-samples", str(args.max_samples),
            "--workers", str(args.workers),
        ]
        if args.cache_dir:
            dl_cmd.extend(["--cache-dir", args.cache_dir])
        run_cmd(
            dl_cmd,
            f"Downloading {args.modality} datasets ({args.max_samples} samples/dataset, {args.workers} workers)",
            cache_dir=args.cache_dir,
        )

        verify_cmd = [sys.executable, "prepare.py", "--verify", "--modality", args.modality]
        if args.cache_dir:
            verify_cmd.extend(["--cache-dir", args.cache_dir])
        run_cmd(verify_cmd, "Verifying cache", cache_dir=args.cache_dir)

    if args.download_only:
        print("\nDownload complete. Exiting (--download-only).")
        return

    # ── Step 2: Train ──
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.log_file:
        log_path = Path(args.log_file)
    else:
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        log_path = runs_dir / f"full_{args.modality}_{ts}.log"

    train_script = f"train_{args.modality}.py"
    train_cmd = [sys.executable, train_script, "--time-budget", str(time_budget_seconds)]
    if args.model:
        train_cmd.extend(["--model", args.model])
    if args.lr is not None:
        train_cmd.extend(["--lr", str(args.lr)])
    if args.batch_size is not None:
        train_cmd.extend(["--batch-size", str(args.batch_size)])
    if args.cache_dir:
        train_cmd.extend(["--cache-dir", args.cache_dir])

    train_env = None
    if args.cache_dir:
        train_env = os.environ.copy()
        train_env["DFRESEARCH_CACHE"] = str(Path(args.cache_dir).expanduser())

    print(f"\n{'=' * 60}")
    print(f"  Training for {args.hours}h ({time_budget_seconds}s)")
    print(f"  $ {' '.join(train_cmd)}")
    print(f"  Logging to: {log_path}")
    print(f"{'=' * 60}\n")

    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            train_cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=train_env,
        )

    # Print the summary from the log
    print(f"\nTraining finished (exit code: {proc.returncode})")
    log_text = log_path.read_text()

    # Extract and display the summary block
    in_summary = False
    for line in log_text.splitlines():
        if line.strip() == "---":
            in_summary = True
        if in_summary:
            print(line)

    if proc.returncode != 0:
        print(f"\nWARNING: Training exited with code {proc.returncode}")
        print(f"Check log: {log_path}")
        # Show last 20 lines for debugging
        lines = log_text.splitlines()
        if len(lines) > 20:
            print("\nLast 20 lines of log:")
            for line in lines[-20:]:
                print(f"  {line}")

    # ── Step 3: Evaluate ──
    if proc.returncode == 0:
        eval_cmd = [sys.executable, "evaluate.py", "--modality", args.modality]
        if args.model:
            eval_cmd.extend(["--model", args.model])
        if args.cache_dir:
            eval_cmd.extend(["--cache-dir", args.cache_dir])
        run_cmd(eval_cmd, "Evaluating trained model", check=False, cache_dir=args.cache_dir)

    # ── Step 4: Export ──
    if proc.returncode == 0 and not args.skip_export:
        model_name = args.model
        if model_name is None:
            # Read the default from the training script
            defaults = {
                "image": "efficientnet-b4",
                "video": "r3d-18",
                "audio": "wav2vec2",
                "image_human": "deepfake-detector-v2",
            }
            model_name = defaults[args.modality]

        export_cmd = [
            sys.executable, "export.py",
            "--modality", args.modality,
            "--model", model_name,
        ]
        run_cmd(export_cmd, "Exporting for competition submission", check=False)

    elapsed = time.time() - t_start
    hours = elapsed / 3600
    print(f"\n{'#' * 60}")
    print(f"#  Done! Total wall time: {hours:.1f}h ({elapsed:.0f}s)")
    print(f"#  Log: {log_path}")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()

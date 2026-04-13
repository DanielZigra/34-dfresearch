"""CLI entry point for dfresearch."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _env_with_cache_dir(cache_dir: str | None):
    if not cache_dir:
        return None
    env = os.environ.copy()
    env["DFRESEARCH_CACHE"] = str(Path(cache_dir).expanduser())
    return env


def main():
    parser = argparse.ArgumentParser(
        prog="dfresearch",
        description="Autonomous deepfake detection research for BitMind Subnet 34",
    )
    subparsers = parser.add_subparsers(dest="command")

    # prepare
    prep = subparsers.add_parser("prepare", help="Download and prepare datasets")
    prep.add_argument("--modality", choices=["image", "video", "audio", "image_human", "all"], default="all")
    prep.add_argument("--verify", action="store_true")
    prep.add_argument("--workers", type=int, default=4)
    prep.add_argument("--max-samples", type=int, default=500)
    prep.add_argument("--refresh-configs", action="store_true")
    prep.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Directory for datasets and config cache (default: ~/.cache/dfresearch)",
    )

    # train
    train = subparsers.add_parser("train", help="Run training")
    train.add_argument("--modality", required=True, choices=["image", "video", "audio", "image_human"])
    train.add_argument("--model", default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--time-budget", type=int, default=None)
    train.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Directory for datasets and config cache (default: ~/.cache/dfresearch)",
    )

    # evaluate
    ev = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    ev.add_argument("--modality", required=True, choices=["image", "video", "audio", "image_human"])
    ev.add_argument("--model", default=None)
    ev.add_argument("--weights", default=None)
    ev.add_argument("--batch-size", type=int, default=None)
    ev.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Directory for datasets and config cache (default: ~/.cache/dfresearch)",
    )

    # export
    exp = subparsers.add_parser("export", help="Export model for competition")
    exp.add_argument("--modality", required=True, choices=["image", "video", "audio", "image_human"])
    exp.add_argument("--model", required=True)
    exp.add_argument("--checkpoint-dir", default=None)
    exp.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    if args.command == "prepare":
        cmd = [sys.executable, str(PROJECT_ROOT / "prepare.py"),
               "--modality", args.modality, "--workers", str(args.workers),
               "--max-samples", str(args.max_samples)]
        if args.verify:
            cmd.append("--verify")
        if args.refresh_configs:
            cmd.append("--refresh-configs")
        if args.cache_dir:
            cmd.extend(["--cache-dir", args.cache_dir])
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=_env_with_cache_dir(args.cache_dir))

    elif args.command == "train":
        script = PROJECT_ROOT / f"train_{args.modality}.py"
        cmd = [sys.executable, str(script)]
        if args.model:
            cmd.extend(["--model", args.model])
        if args.lr is not None:
            cmd.extend(["--lr", str(args.lr)])
        if args.batch_size is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])
        if args.time_budget is not None:
            cmd.extend(["--time-budget", str(args.time_budget)])
        if args.cache_dir:
            cmd.extend(["--cache-dir", args.cache_dir])
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=_env_with_cache_dir(args.cache_dir))

    elif args.command == "evaluate":
        cmd = [sys.executable, str(PROJECT_ROOT / "evaluate.py"),
               "--modality", args.modality]
        if args.model:
            cmd.extend(["--model", args.model])
        if args.weights:
            cmd.extend(["--weights", args.weights])
        if args.batch_size is not None:
            cmd.extend(["--batch-size", str(args.batch_size)])
        if args.cache_dir:
            cmd.extend(["--cache-dir", args.cache_dir])
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=_env_with_cache_dir(args.cache_dir))

    elif args.command == "export":
        cmd = [sys.executable, str(PROJECT_ROOT / "export.py"),
               "--modality", args.modality, "--model", args.model]
        if args.checkpoint_dir:
            cmd.extend(["--checkpoint-dir", args.checkpoint_dir])
        if args.output_dir:
            cmd.extend(["--output-dir", args.output_dir])
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
export.py — Export trained models to gasbench-compatible safetensors ZIP format.

Packages a trained model into the exact submission format required by
BitMind Subnet 34:

    my_detector.zip
    ├── model_config.yaml
    ├── model.py
    └── model.safetensors

Usage:
    uv run export.py --modality image --model efficientnet-b4
    uv run export.py --modality video --model r3d-18
    uv run export.py --modality audio --model wav2vec2
    uv run export.py --modality image --model efficientnet-b4 --weights path/to/model.safetensors
"""

import argparse
import zipfile
from pathlib import Path

import yaml

# Maps model names to their source module paths for the model.py export
MODEL_MODULES = {
    "image": {
        "efficientnet-b4": "dfresearch.models.image.efficientnet",
        "clip-vit-l14": "dfresearch.models.image.clip_vit",
        "smogy-swin": "dfresearch.models.image.smogy_swin",
        "siglip-dinov2-ensemble": "dfresearch.models.bombek_siglip_dinov2_export",
    },
    "video": {
        "r3d-18": "dfresearch.models.video.r3d",
        "videomae": "dfresearch.models.video.videomae",
        "hiera": "dfresearch.models.video.hiera",
    },
    "audio": {
        "wav2vec2": "dfresearch.models.audio.wav2vec2",
        "ast": "dfresearch.models.audio.ast_model",
        "wavlm": "dfresearch.models.audio.wavlm",
    },
    "image_human": {
        "deepfake-detector-v2": "dfresearch.models.image_human.deep_fake_detector_v2",
        "airealnet": "dfresearch.models.image_human.airealnet",
        "real-vs-ai-face": "dfresearch.models.image_human.real_vs_ai_face",
        "human-faces-ai-vs-real": "dfresearch.models.image_human.human_faces_ai_vs_real",
        "siglip-dinov2-ensemble": "dfresearch.models.bombek_siglip_dinov2_export",
    },
}

# Default preprocessing configs per modality
PREPROCESSING_CONFIGS = {
    "image": {
        "resize": [224, 224],
    },
    "video": {
        "resize": [224, 224],
        "num_frames": 16,
    },
    "audio": {
        "sample_rate": 16000,
        "duration_seconds": 6.0,
    },
    "image_human": {
        "resize": [224, 224],
    },
}


def generate_model_config(modality: str, model_name: str) -> dict:
    """Generate model_config.yaml content."""
    return {
        "name": f"dfresearch-{model_name}",
        "version": "1.0.0",
        "modality": modality,
        "preprocessing": PREPROCESSING_CONFIGS[modality],
        "model": {
            "num_classes": 2,
            "weights_file": "model.safetensors",
        },
    }


def generate_model_py(modality: str, model_name: str) -> str:
    """
    Return the model source file for submission.

    Each model module is already self-contained with a load_model() function,
    so this just reads and returns the source file.
    """
    module_name = MODEL_MODULES[modality][model_name]
    module_path = Path("src") / module_name.replace(".", "/")
    source_file = module_path.with_suffix(".py")

    if source_file.exists():
        return source_file.read_text()

    raise FileNotFoundError(
        f"Model source not found: {source_file}\n"
        f"Expected module: {module_name}"
    )


def export_model(
    modality: str,
    model_name: str,
    checkpoint_dir: Path | None = None,
    output_dir: Path = Path("results/exports"),
) -> Path:
    """
    Export a trained model to a gasbench-compatible ZIP.

    The checkpoint directory should already contain model.safetensors,
    model.py, and model_config.yaml (written by train_*.py). If any
    are missing, they are generated on the fly.

    Args:
        modality: "image", "video", or "audio".
        model_name: Model identifier (e.g., "efficientnet-b4").
        checkpoint_dir: Directory with model files. Defaults to results/checkpoints/{modality}.
        output_dir: Directory to write the ZIP file.

    Returns:
        Path to the output ZIP file.
    """
    if checkpoint_dir is None:
        checkpoint_dir = Path("results/checkpoints") / f"{modality}_{model_name}"

    weights = checkpoint_dir / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            f"Run train_{modality}.py first to generate a checkpoint."
        )

    if not (checkpoint_dir / "model.py").exists():
        (checkpoint_dir / "model.py").write_text(generate_model_py(modality, model_name))

    if not (checkpoint_dir / "model_config.yaml").exists():
        config = generate_model_config(modality, model_name)
        with open(checkpoint_dir / "model_config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_name = f"{modality}_detector_{model_name}.zip"
    zip_path = output_dir / zip_name

    required_files = ["model.safetensors", "model.py", "model_config.yaml"]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in required_files:
            f = checkpoint_dir / name
            if f.exists():
                zf.write(f, name)

    print(f"Exported: {zip_path}")
    print(f"  Contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"    {info.filename:<30} {info.file_size:>10} bytes")

    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="Export trained model to gasbench-compatible ZIP"
    )
    parser.add_argument(
        "--modality",
        required=True,
        choices=["image", "video", "audio", "image_human"],
        help="Model modality",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., efficientnet-b4, clip-vit-l14, r3d-18, videomae, wav2vec2, ast)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory (default: results/checkpoints/{modality}_{model})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exports"),
        help="Output directory (default: results/exports)",
    )
    args = parser.parse_args()

    if args.modality not in MODEL_MODULES:
        parser.error(f"Unknown modality: {args.modality}")
    if args.model not in MODEL_MODULES[args.modality]:
        available = list(MODEL_MODULES[args.modality].keys())
        parser.error(f"Unknown {args.modality} model: {args.model}. Available: {available}")

    zip_path = export_model(
        modality=args.modality,
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
    )

    print(f"\nTo test locally with gasbench:")
    print(f"  gasbench run --{args.modality}-model {zip_path} --small")
    print(f"\nTo push to BitMind Subnet 34:")
    print(f"  gascli d push --{args.modality}-model {zip_path} --wallet-name <NAME> --wallet-hotkey <KEY>")


if __name__ == "__main__":
    main()

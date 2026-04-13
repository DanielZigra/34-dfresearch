#!/usr/bin/env python3
"""
Fold LoRA (SigLIP peft + DINOv2 LoRALinear) into dense weights for peft-free export.

Requires: peft, timm, transformers (same as training).

Usage:
    python scripts/merge_bombek_for_export.py \\
        --input ~/.cache/huggingface/hub/.../pytorch_model.pt \\
        --output results/checkpoints/image_siglip-dinov2-ensemble/model.safetensors

Optionally copies export ``model.py`` next to output if ``--write-model-py`` is set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

# Project root: dfresearch/
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _loralinear_to_dense(m: nn.Module) -> nn.Linear:
    from dfresearch.models.image.siglip_dinov2_ensemble import LoRALinear

    if not isinstance(m, LoRALinear):
        raise TypeError("expected LoRALinear")
    W = m.original.weight.data.clone()
    delta = (m.lora_B.weight @ m.lora_A.weight) * m.scaling
    W = W + delta
    linear = nn.Linear(m.original.in_features, m.original.out_features, bias=m.original.bias is not None)
    linear.weight.data = W
    if m.original.bias is not None:
        linear.bias.data = m.original.bias.data.clone()
    return linear


def _merge_dinov2_loras(dino: nn.Module) -> None:
    from dfresearch.models.image.siglip_dinov2_ensemble import LoRALinear

    for module in dino.modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, LoRALinear):
            module.qkv = _loralinear_to_dense(module.qkv)


def merge_checkpoint(input_path: Path) -> tuple[dict, dict]:
    """Load LoRA checkpoint, return (state_dict, meta) for dense export."""
    from safetensors.torch import load_file

    from dfresearch.models.image.siglip_dinov2_ensemble import _create_model_with_lora

    if input_path.suffix == ".safetensors":
        full_sd = load_file(str(input_path))
        meta_side = input_path.with_name("bombek_meta.json")
        cfg = json.loads(meta_side.read_text()) if meta_side.exists() else {}
        inner_sd = {k[6:]: v for k, v in full_sd.items() if k.startswith("model.")}
        if not inner_sd:
            raise ValueError(
                f"No keys prefixed with 'model.' in {input_path}; expected a training checkpoint."
            )
    else:
        ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {}) or {}
        inner_sd = ckpt["model_state_dict"]

    inner = _create_model_with_lora(
        siglip_model_name=cfg.get("siglip_model", "google/siglip2-so400m-patch14-384"),
        dinov2_model_name=cfg.get("dinov2_model", "vit_large_patch14_dinov2.lvd142m"),
        image_size=int(cfg.get("image_size", 392)),
        lora_rank=int(cfg.get("lora_rank", 32)),
        lora_alpha=int(cfg.get("lora_alpha", 64)),
        lora_dropout=float(cfg.get("lora_dropout", 0.1)),
    )
    inner.load_state_dict(inner_sd)

    if hasattr(inner.siglip, "merge_and_unload"):
        inner.siglip = inner.siglip.merge_and_unload()
    _merge_dinov2_loras(inner.dinov2)

    import copy

    from dfresearch.models.bombek_siglip_dinov2_export import (
        EnsembleAIDetector,
        SigLIPDinov2EnsembleDetector,
    )

    dense = EnsembleAIDetector(
        cfg.get("siglip_model", "google/siglip2-so400m-patch14-384"),
        cfg.get("dinov2_model", "vit_large_patch14_dinov2.lvd142m"),
        int(cfg.get("image_size", 392)),
    )
    dense.siglip = inner.siglip
    dense.dinov2 = inner.dinov2
    dense.classifier = copy.deepcopy(inner.classifier)

    wrapper = SigLIPDinov2EnsembleDetector(
        num_classes=2,
        pretrained=False,
        siglip_model_name=cfg.get("siglip_model", "google/siglip2-so400m-patch14-384"),
        dinov2_model_name=cfg.get("dinov2_model", "vit_large_patch14_dinov2.lvd142m"),
        image_size=int(cfg.get("image_size", 392)),
    )
    wrapper.model = dense

    meta = {
        "image_size": int(cfg.get("image_size", 392)),
        "siglip_model": cfg.get("siglip_model", "google/siglip2-so400m-patch14-384"),
        "dinov2_model": cfg.get("dinov2_model", "vit_large_patch14_dinov2.lvd142m"),
    }
    return wrapper.state_dict(), meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Bombek pytorch_model.pt or compatible ckpt")
    p.add_argument("--output", type=Path, required=True, help="Output model.safetensors")
    p.add_argument("--meta-output", type=Path, default=None, help="Write bombek_meta.json (default: next to --output)")
    p.add_argument("--write-model-py", action="store_true", help="Write peft-free model.py next to --output")
    args = p.parse_args()

    sd, meta = merge_checkpoint(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, args.output)
    print(f"Wrote merged weights: {args.output}")

    meta_path = args.meta_output or args.output.with_name("bombek_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote meta: {meta_path}")

    if args.write_model_py:
        export_src = _ROOT / "src" / "dfresearch" / "models" / "bombek_siglip_dinov2_export.py"
        out_py = args.output.with_name("model.py")
        out_py.write_text(export_src.read_text())
        print(f"Wrote {out_py}")


if __name__ == "__main__":
    main()

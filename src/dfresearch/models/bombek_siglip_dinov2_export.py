"""
Peft-free inference for Bombek1 SigLIP2 + DINOv2 ensemble.

Weights must be **merged** (LoRA folded into base layers). Produce ``model.safetensors`` with::

    python scripts/merge_bombek_for_export.py --input <pytorch_model.pt|model.safetensors> --output model.safetensors

Training / LoRA loading uses ``image.siglip_dinov2_ensemble`` (requires peft).

Input:  [B, 3, H, W] uint8 [0, 255]
Output: [B, 2] logits [real, synthetic]
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID = "Bombek1/ai-image-detector-siglip-dinov2"
MODEL_FILENAME = "pytorch_model.pt"

SIGLIP_SIZE = 384
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class EnsembleAIDetector(nn.Module):
    """Dense SigLIP + DINOv2 + head (merged weights; no LoRA, no peft)."""

    def __init__(self, siglip_model_name: str, dinov2_model_name: str, image_size: int = 392):
        super().__init__()

        import timm
        from transformers import SiglipVisionModel

        self.siglip = SiglipVisionModel.from_pretrained(siglip_model_name, torch_dtype=torch.bfloat16)
        self.siglip_dim = self.siglip.config.hidden_size

        self.dinov2 = timm.create_model(
            dinov2_model_name,
            pretrained=True,
            num_classes=0,
            img_size=image_size,
        )
        self.dinov2_dim = self.dinov2.num_features

        self.classifier = ClassificationHead(self.siglip_dim + self.dinov2_dim)

    def forward(self, siglip_pixels: torch.Tensor, dinov2_pixels: torch.Tensor) -> torch.Tensor:
        siglip_features = self.siglip(pixel_values=siglip_pixels).pooler_output
        dinov2_features = self.dinov2(dinov2_pixels)
        combined = torch.cat([siglip_features.float(), dinov2_features], dim=-1)
        return self.classifier(combined)


class SigLIPDinov2EnsembleDetector(nn.Module):
    """Wrapper with dfresearch [real, synthetic] logits; inner weights are dense."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.0,
        siglip_model_name: str = "google/siglip2-so400m-patch14-384",
        dinov2_model_name: str = "vit_large_patch14_dinov2.lvd142m",
        image_size: int = 392,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("This checkpoint is binary; use num_classes=2.")
        if pretrained:
            raise RuntimeError(
                "Use merge_bombek_for_export.py to build merged weights; this module has no peft/LoRA loader."
            )

        self.image_size = image_size
        self.model = EnsembleAIDetector(siglip_model_name, dinov2_model_name, image_size)

        self.register_buffer("siglip_mean", torch.tensor(SIGLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("siglip_std", torch.tensor(SIGLIP_STD).view(1, 3, 1, 1))
        self.register_buffer("dinov2_mean", torch.tensor(DINOV2_MEAN).view(1, 3, 1, 1))
        self.register_buffer("dinov2_std", torch.tensor(DINOV2_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0

        siglip = F.interpolate(x, size=(SIGLIP_SIZE, SIGLIP_SIZE), mode="bilinear", align_corners=False)
        siglip = (siglip - self.siglip_mean) / self.siglip_std

        dinov2 = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        dinov2 = (dinov2 - self.dinov2_mean) / self.dinov2_std

        with torch.amp.autocast("cuda", enabled=(x.device.type == "cuda")):
            ai_logit = self.model(siglip, dinov2)

        real_logit = torch.zeros_like(ai_logit)
        return torch.stack([real_logit, ai_logit], dim=1)


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load merged ``model.safetensors`` (full ``SigLIPDinov2EnsembleDetector`` state dict)."""
    from safetensors.torch import load_file

    wp = Path(weights_path)
    meta_path = wp.with_name("bombek_meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        model = SigLIPDinov2EnsembleDetector(
            num_classes=num_classes,
            pretrained=False,
            siglip_model_name=meta.get("siglip_model", "google/siglip2-so400m-patch14-384"),
            dinov2_model_name=meta.get("dinov2_model", "vit_large_patch14_dinov2.lvd142m"),
            image_size=int(meta.get("image_size", 392)),
        )
    else:
        model = SigLIPDinov2EnsembleDetector(num_classes=num_classes, pretrained=False)

    sd = load_file(weights_path)
    model.load_state_dict(sd, strict=True)
    model.train(False)
    return model

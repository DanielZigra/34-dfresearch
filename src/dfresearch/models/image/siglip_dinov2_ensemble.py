"""
SigLIP2 + DINOv2 ensemble detector from Bombek1/ai-image-detector-siglip-dinov2.

The upstream checkpoint outputs a single logit for P(AI). dfresearch expects 2-class logits
ordered as [real, synthetic], so we convert via softmax([0, logit]) == sigmoid(logit).

Input:  [B, 3, H, W] uint8 [0, 255]
Output: [B, 2] logits [real, synthetic]
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID = "Bombek1/ai-image-detector-siglip-dinov2"
MODEL_FILENAME = "pytorch_model.pt"

# SigLIP2 preprocessor_config.json (google/siglip2-so400m-patch14-384)
SIGLIP_SIZE = 384
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# DINOv2 ImageNet normalization
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)


class LoRALinear(nn.Module):
    """Custom LoRA for DINOv2 qkv projection (matches upstream model.py)."""

    def __init__(self, original: nn.Linear, rank: int, alpha: float, dropout: float = 0.1):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank

        for p in self.original.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(original.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


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
        logits = self.classifier(combined)
        return logits


def _create_model_with_lora(
    siglip_model_name: str = "google/siglip2-so400m-patch14-384",
    dinov2_model_name: str = "vit_large_patch14_dinov2.lvd142m",
    image_size: int = 392,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
) -> EnsembleAIDetector:
    from peft import LoraConfig, get_peft_model

    model = EnsembleAIDetector(siglip_model_name, dinov2_model_name, image_size)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    model.siglip = get_peft_model(model.siglip, lora_config)

    for _, module in model.dinov2.named_modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, lora_rank, lora_alpha, lora_dropout)

    return model


def _download_checkpoint(cache_dir: Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    if cache_dir is not None:
        cache_dir = Path(cache_dir)

    return Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_FILENAME,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )


class SigLIPDinov2EnsembleDetector(nn.Module):
    """
    Bombek1 SigLIP2+DINOv2 ensemble as a dfresearch-compatible 2-class detector.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        if num_classes != 2:
            raise ValueError("This checkpoint is binary; use num_classes=2.")

        # Load checkpoint (contains config + LoRA weights + head weights)
        ckpt = None
        cfg: dict = {}
        if pretrained:
            from dfresearch.data import get_cache_dir

            model_cache = get_cache_dir() / "models"
            model_cache.mkdir(parents=True, exist_ok=True)
            ckpt_path = _download_checkpoint(cache_dir=model_cache)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {}) or {}

        self.image_size = int(cfg.get("image_size", 392))

        self.model = _create_model_with_lora(
            siglip_model_name=cfg.get("siglip_model", "google/siglip2-so400m-patch14-384"),
            dinov2_model_name=cfg.get("dinov2_model", "vit_large_patch14_dinov2.lvd142m"),
            image_size=self.image_size,
            lora_rank=int(cfg.get("lora_rank", 32)),
            lora_alpha=int(cfg.get("lora_alpha", 64)),
            lora_dropout=float(cfg.get("lora_dropout", 0.1)),
        )

        if ckpt is not None:
            self.model.load_state_dict(ckpt["model_state_dict"])

        self.register_buffer("siglip_mean", torch.tensor(SIGLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("siglip_std", torch.tensor(SIGLIP_STD).view(1, 3, 1, 1))
        self.register_buffer("dinov2_mean", torch.tensor(DINOV2_MEAN).view(1, 3, 1, 1))
        self.register_buffer("dinov2_std", torch.tensor(DINOV2_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: uint8 in [0,255]
        x = x.float() / 255.0

        # SigLIP expects 384x384, normalized to mean/std=0.5 after rescale.
        siglip = F.interpolate(x, size=(SIGLIP_SIZE, SIGLIP_SIZE), mode="bilinear", align_corners=False)
        siglip = (siglip - self.siglip_mean) / self.siglip_std

        # DINOv2 branch uses image_size from checkpoint config (default 392) and ImageNet norm.
        dinov2 = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        dinov2 = (dinov2 - self.dinov2_mean) / self.dinov2_std

        # Use autocast on CUDA, matching upstream usage.
        with torch.amp.autocast("cuda", enabled=(x.device.type == "cuda")):
            ai_logit = self.model(siglip, dinov2)

        real_logit = torch.zeros_like(ai_logit)
        return torch.stack([real_logit, ai_logit], dim=1)


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """
    Load from a local Bombek1 `pytorch_model.pt` checkpoint (gasbench-style entry point).
    """
    model = SigLIPDinov2EnsembleDetector(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.train(False)
    return model


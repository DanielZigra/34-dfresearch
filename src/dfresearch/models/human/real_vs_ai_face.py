"""
ViT from ruhul590/real-vs-ai-human-face-classify (ViT-B/16, 224).

Config id2label uses placeholder names; treat as binary real vs non-real face.
Assumed mapping: 0 = real / normal, 1 = synthetic (matches typical face-AI setups).

Input:  [B, 3, 224, 224] uint8 [0, 255]
Output: [B, 2] logits [real, synthetic]
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file

MODEL_ID = "ruhul590/real-vs-ai-human-face-classify"

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class RealVsAIFaceDetector(nn.Module):
    """ViT binary face classifier."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        if num_classes != 2:
            raise ValueError("This checkpoint is 2-class; use num_classes=2.")

        from transformers import AutoConfig, AutoModelForImageClassification

        if pretrained:
            self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        else:
            config = AutoConfig.from_pretrained(MODEL_ID)
            self.model = AutoModelForImageClassification.from_config(config)

        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        return self.model(pixel_values=x).logits


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load model with pretrained weights (gasbench entry point)."""
    model = RealVsAIFaceDetector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model

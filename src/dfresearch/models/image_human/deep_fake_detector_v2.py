"""
ViT classifier from prithivMLmods/Deep-Fake-Detector-v2-Model (ViT-B/16, 224).

HF id2label: 0 = Realism, 1 = Deepfake — matches dfresearch [real, synthetic].

Input:  [B, 3, 224, 224] uint8 [0, 255]
Output: [B, 2] logits [real, synthetic]
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file

MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"

# Matches preprocessor_config.json on the Hub
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class DeepFakeDetectorV2Detector(nn.Module):
    """ViT binary detector with Deep-Fake-Detector-v2 weights."""

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
    model = DeepFakeDetectorV2Detector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model

"""
ViT from dima806/human_faces_ai_vs_real_image_detection (ViT-B/16, 224).

HF id2label: 0 = AI-Generated, 1 = Real — dfresearch uses 0 = real, 1 = synthetic, so logits are reordered.

Input:  [B, 3, 224, 224] uint8 [0, 255]
Output: [B, 2] logits [real, synthetic]
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file

MODEL_ID = "dima806/human_faces_ai_vs_real_image_detection"

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class HumanFacesAIvsRealDetector(nn.Module):
    """ViT binary detector with swapped output order to match [real, synthetic]."""

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
        logits = self.model(pixel_values=x).logits
        # HF: 0 = AI (synthetic), 1 = real
        return logits[:, [1, 0]]


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load model with pretrained weights (gasbench entry point)."""
    model = HumanFacesAIvsRealDetector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model

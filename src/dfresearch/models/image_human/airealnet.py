"""
Swin Transformer v2 from Modotte/AIRealNet (trained at 256×256; train/eval here at 224×224).

HF id2label: 0 = artificial, 1 = real — dfresearch uses 0 = real, 1 = synthetic, so logits are reordered.

Input:  [B, 3, 224, 224] uint8 [0, 255]
Output: [B, 2] logits [real, synthetic]
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file

MODEL_ID = "Modotte/AIRealNet"

# ImageNet normalization from preprocessor_config.json on the Hub
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AIRealNetDetector(nn.Module):
    """Swinv2 binary detector; uses interpolate_pos_encoding for 224 inputs vs 256 pretrain."""

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

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        logits = self.model(pixel_values=x, interpolate_pos_encoding=True).logits
        # HF: index 0 = artificial (synthetic), 1 = real
        return logits[:, [1, 0]]


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load model with pretrained weights (gasbench entry point)."""
    model = AIRealNetDetector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model

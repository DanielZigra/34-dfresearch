"""
SMOGY Swin Transformer for image deepfake detection.

Uses the Smogy/SMOGY-Ai-images-detector model — a Swin Transformer already
fine-tuned for AI-generated image detection (98.2% accuracy on test split).
Unlike other baselines that start from ImageNet/CLIP weights, this model
arrives pre-trained for the forensics task.

License: CC-BY-NC-4.0 (non-commercial use only)

Input:  [B, 3, 224, 224] uint8 [0, 255]
Output: [B, 2] logits [real, synthetic]
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SMOGYSwinDetector(nn.Module):
    """SMOGY Swin Transformer binary deepfake detector."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = "Smogy/SMOGY-Ai-images-detector",
        dropout: float = 0.2,
    ):
        super().__init__()

        if pretrained:
            from transformers import AutoModelForImageClassification
            hf_model = AutoModelForImageClassification.from_pretrained(model_name)
            self.swin = hf_model.swin
            self.feat_dim = hf_model.classifier.in_features
        else:
            from transformers import SwinModel, SwinConfig
            config = SwinConfig(
                image_size=224,
                patch_size=4,
                num_channels=3,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
            )
            self.swin = SwinModel(config)
            self.feat_dim = config.hidden_size  # typically 1024

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes),
        )

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] uint8 [0, 255]
        x = x.float() / 255.0
        x = (x - self.mean) / self.std

        outputs = self.swin(pixel_values=x)
        pooled = outputs.pooler_output  # [B, feat_dim]

        return self.head(pooled)  # [B, 2]


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load model with pretrained weights (gasbench entry point)."""
    model = SMOGYSwinDetector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model

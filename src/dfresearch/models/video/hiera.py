"""
Hiera baseline for video deepfake detection.

Uses Meta's hierarchical vision transformer pretrained with MAE.
Fast, modern architecture that learns spatial biases through pretraining
rather than hand-designed modules (unlike Swin/MViT).

Processes video as individual frames through the image encoder, then
pools frame-level features for temporal aggregation.

Input:  [B, T, 3, H, W] uint8 [0, 255]  (T = num_frames, typically 16)
Output: [B, 2] logits [real, synthetic]
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HieraDetector(nn.Module):
    """Hiera hierarchical ViT binary deepfake detector."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = "facebook/hiera-base-224-hf",
        dropout: float = 0.3,
    ):
        super().__init__()

        if pretrained:
            from transformers import HieraModel
            self.encoder = HieraModel.from_pretrained(model_name)
        else:
            from transformers import HieraModel, HieraConfig
            config = HieraConfig(
                embed_dim=96,
                num_heads=[1, 2, 4, 8],
                depths=[2, 3, 16, 3],
                image_size=[224, 224],
                patch_size=[7, 7],
                patch_stride=[4, 4],
                patch_padding=[3, 3],
            )
            self.encoder = HieraModel(config)

        self.feat_dim = self.encoder.config.embed_dim * 8  # 768 for base

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
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
        # x: [B, T, 3, H, W] uint8 [0, 255]
        batch_size, num_frames = x.shape[:2]

        # Process all frames through the encoder
        x = x.view(batch_size * num_frames, 3, x.shape[3], x.shape[4])
        x = x.float() / 255.0
        x = (x - self.mean) / self.std

        outputs = self.encoder(pixel_values=x)
        pooled = outputs.pooler_output  # [B*T, feat_dim]

        # Reshape back and aggregate across frames
        pooled = pooled.view(batch_size, num_frames, self.feat_dim)  # [B, T, D]
        pooled = pooled.permute(0, 2, 1)  # [B, D, T]
        pooled = self.temporal_pool(pooled).squeeze(-1)  # [B, D]

        return self.head(pooled)  # [B, 2]


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load model with pretrained weights (gasbench entry point)."""
    model = HieraDetector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model

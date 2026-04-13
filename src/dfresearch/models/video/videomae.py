"""
VideoMAE baseline for video deepfake detection.

Uses MCG-NJU's VideoMAE-base pretrained on Kinetics-400 via masked autoencoding.
Strong self-supervised spatiotemporal features with excellent transfer learning.

Input:  [B, T, 3, H, W] uint8 [0, 255]  (T = num_frames, typically 16)
Output: [B, 2] logits [real, synthetic]
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VideoMAEDetector(nn.Module):
    """VideoMAE binary deepfake detector."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = "MCG-NJU/videomae-base",
        dropout: float = 0.3,
    ):
        super().__init__()

        if pretrained:
            from transformers import VideoMAEModel
            self.encoder = VideoMAEModel.from_pretrained(model_name)
        else:
            from transformers import VideoMAEModel, VideoMAEConfig
            config = VideoMAEConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                num_frames=16,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            self.encoder = VideoMAEModel(config)

        self.feat_dim = self.encoder.config.hidden_size  # 768

        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes),
        )

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 3, H, W] uint8 [0, 255]
        x = x.float() / 255.0
        x = (x - self.mean) / self.std

        # VideoMAE expects pixel_values: [B, T, C, H, W]
        outputs = self.encoder(pixel_values=x)
        # Mean pool over sequence tokens
        sequence_output = outputs.last_hidden_state  # [B, seq_len, hidden]
        pooled = sequence_output.mean(dim=1)         # [B, hidden]

        return self.head(pooled)  # [B, 2]


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load model with pretrained weights (gasbench entry point)."""
    model = VideoMAEDetector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model

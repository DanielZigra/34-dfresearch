"""
R3D-18 baseline for video deepfake detection.

Uses torchvision's Kinetics-pretrained 3D ResNet-18 with a 2-class head.
3D convolutions capture temporal artifacts between frames that are
characteristic of video deepfakes.

Input:  [B, T, 3, H, W] uint8 [0, 255]  (T = num_frames, typically 16)
Output: [B, 2] logits [real, synthetic]
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from safetensors.torch import load_file


KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


class R3D18Detector(nn.Module):
    """3D ResNet-18 binary deepfake detector."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        if pretrained:
            backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        else:
            backbone = r3d_18(weights=None)

        self.feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes),
        )

        self.register_buffer(
            "mean", torch.tensor(KINETICS_MEAN).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(KINETICS_STD).view(1, 1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 3, H, W] uint8 [0, 255]
        x = x.float() / 255.0
        x = (x - self.mean) / self.std

        # R3D expects [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        features = self.backbone(x)  # [B, feat_dim]
        return self.head(features)   # [B, 2]


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load model with pretrained weights (gasbench entry point)."""
    model = R3D18Detector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model

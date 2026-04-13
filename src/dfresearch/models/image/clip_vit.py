"""
CLIP ViT-L/14 baseline for image deepfake detection.

Fine-tunes OpenAI's CLIP vision encoder for binary deepfake classification.
Research shows CLIP features generalize exceptionally well across unseen
generators (UniversalFakeDetect, CVPR 2023).

Input:  [B, 3, 224, 224] uint8 [0, 255]
Output: [B, 2] logits [real, synthetic]
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPViTDetector(nn.Module):
    """CLIP ViT-L/14 binary deepfake detector."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        model_name: str = "openai/clip-vit-large-patch14",
    ):
        super().__init__()

        if pretrained:
            from transformers import CLIPVisionModel
            self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        else:
            from transformers import CLIPVisionModel, CLIPVisionConfig
            config = CLIPVisionConfig(
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=24,
                num_attention_heads=16,
                image_size=224,
                patch_size=14,
            )
            self.vision_model = CLIPVisionModel(config)

        self.feat_dim = self.vision_model.config.hidden_size  # 1024 for ViT-L

        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim, num_classes),
        )

        self.register_buffer(
            "mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] uint8 [0, 255]
        x = x.float() / 255.0
        x = (x - self.mean) / self.std

        outputs = self.vision_model(pixel_values=x)
        pooled = outputs.pooler_output  # [B, 1024]

        return self.head(pooled)  # [B, 2]


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load model with pretrained weights (gasbench entry point)."""
    model = CLIPViTDetector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model

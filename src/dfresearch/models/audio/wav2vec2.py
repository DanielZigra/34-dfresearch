"""
Wav2Vec2 baseline for audio deepfake detection.

Fine-tunes facebook/wav2vec2-base for binary spoofing classification.
Dominant approach in ASVspoof challenges — self-supervised pretrained on
960h of speech, strong out-of-the-box features for detecting synthesis artifacts.

Input:  [B, 96000] float32 [-1, 1]  (16kHz, 6 seconds)
Output: [B, 2] logits [real, synthetic]
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file


class Wav2Vec2Detector(nn.Module):
    """Wav2Vec2 binary deepfake detector."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = "facebook/wav2vec2-base",
        freeze_feature_encoder: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()

        if pretrained:
            from transformers import Wav2Vec2Model
            self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        else:
            from transformers import Wav2Vec2Model, Wav2Vec2Config
            config = Wav2Vec2Config(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            self.encoder = Wav2Vec2Model(config)

        if freeze_feature_encoder:
            self.encoder.feature_extractor._freeze_parameters()

        self.feat_dim = self.encoder.config.hidden_size  # 768

        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 96000] float32 [-1, 1]
        outputs = self.encoder(input_values=x)
        hidden = outputs.last_hidden_state   # [B, seq_len, 768]
        pooled = hidden.mean(dim=1)          # [B, 768]
        return self.head(pooled)             # [B, 2]


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load model with pretrained weights (gasbench entry point)."""
    model = Wav2Vec2Detector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model

"""
Audio Spectrogram Transformer (AST) baseline for audio deepfake detection.

Applies a Vision Transformer to mel-spectrogram inputs. Pretrained on AudioSet,
provides strong frequency-domain features complementary to Wav2Vec2's time-domain
approach.

Input:  [B, 96000] float32 [-1, 1]  (16kHz, 6 seconds)
Output: [B, 2] logits [real, synthetic]
"""

import math

import torch
import torch.nn as nn
from safetensors.torch import load_file


class ASTDetector(nn.Module):
    """Audio Spectrogram Transformer binary deepfake detector."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        dropout: float = 0.2,
        n_mels: int = 128,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        target_length: int = 1024,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.target_length = target_length

        import torchaudio
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        if pretrained:
            from transformers import ASTModel
            self.encoder = ASTModel.from_pretrained(model_name)
        else:
            from transformers import ASTModel, ASTConfig
            config = ASTConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_length=target_length,
                num_mel_bins=n_mels,
            )
            self.encoder = ASTModel(config)

        self.feat_dim = self.encoder.config.hidden_size

        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 96000] float32 [-1, 1]
        mel = self.mel_transform(x)                        # [B, n_mels, time]
        mel = (mel + 1e-8).log()                           # log-mel
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        # AST expects input_values: [B, max_length, n_mels]
        mel = mel.permute(0, 2, 1)                         # [B, time, n_mels]

        # Pad or truncate to fixed target_length so it matches position embeddings
        time_steps = mel.shape[1]
        if time_steps > self.target_length:
            mel = mel[:, :self.target_length, :]
        elif time_steps < self.target_length:
            pad = torch.zeros(
                mel.shape[0], self.target_length - time_steps, mel.shape[2],
                device=mel.device, dtype=mel.dtype,
            )
            mel = torch.cat([mel, pad], dim=1)

        outputs = self.encoder(input_values=mel)
        pooled = outputs.pooler_output                     # [B, 768]

        return self.head(pooled)                           # [B, 2]


def load_model(weights_path: str, num_classes: int = 2) -> nn.Module:
    """Load model with pretrained weights (gasbench entry point)."""
    model = ASTDetector(num_classes=num_classes, pretrained=False)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.train(False)
    return model

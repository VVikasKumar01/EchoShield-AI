from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class DefenseNoiseClassifier(nn.Module):
    """Ultra-lightweight temporal CNN for acoustic noise regime categorization.

    Categorizes incoming audio frames into:
      Class 0: Stationary noise (engines, field generators, steady drone hum)
      Class 1: Non-stationary / dynamic noise (varying rotor blades, sirens, moving convoy)
      Class 2: Impulsive shock noise (gunfire, mortar/artillery blast, shrapnel)

    Complexity:
      ~15k parameters, < 0.05 MFLOPs per frame. Adds 0 algorithmic latency.
    """

    NUM_CLASSES: int = 3
    CLASS_STATIONARY: int = 0
    CLASS_NON_STATIONARY: int = 1
    CLASS_IMPULSIVE: int = 2

    def __init__(
        self,
        in_channels: int = 32,
        conv_ch: int = 32,
        kernel_size: int = 3,
        num_classes: int = 3,
    ):
        """Initialize the noise classifier.

        Args:
            in_channels (int): Input feature dimension per frame (e.g. 32 ERB bands for 1 mic,
                or 64 for 2 mics). Default is 32.
            conv_ch (int): Internal hidden feature dimension.
            kernel_size (int): Temporal convolution kernel size.
            num_classes (int): Number of target noise classes (3: Stationary, Non-Stat, Impulsive).
        """
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_ch, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(conv_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_ch, conv_ch // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(conv_ch // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(conv_ch // 2, num_classes)

    def forward(self, erb_feat: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Classify noise regime from ERB feature representations.

        Args:
            erb_feat (Tensor): ERB energy features. Can be [B, 1, T, E] or [B, T, E] or [B, E, T].

        Returns:
            logits (Tensor): Raw class logits of shape [B, num_classes].
            probs (Tensor): Softmax probability distribution of shape [B, num_classes].
            confidence (Tensor): Highest class confidence of shape [B, 1].
        """
        # Ensure shape [B, in_channels, T]
        if erb_feat.dim() == 4:
            # [B, 1, T, E] -> [B, E, T]
            b, c, t, e = erb_feat.shape
            x = erb_feat.reshape(b, c * e, t)
        elif erb_feat.dim() == 3:
            # Check if [B, T, E] vs [B, E, T]
            if erb_feat.shape[1] == self.in_channels:
                x = erb_feat
            else:
                x = erb_feat.transpose(1, 2)
        elif erb_feat.dim() == 2:
            # Single frame [B, E] -> [B, E, 1]
            x = erb_feat.unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported ERB tensor shape {erb_feat.shape}")

        if x.shape[1] != self.in_channels:
            # If channels don't match, adapt via projection or slicing
            if x.shape[1] > self.in_channels:
                x = x[:, : self.in_channels, :]
            else:
                # Pad
                pad = torch.zeros(
                    x.shape[0], self.in_channels - x.shape[1], x.shape[2], device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, pad], dim=1)

        feat = self.conv(x).squeeze(-1)  # [B, conv_ch // 2]
        logits = self.fc(feat)  # [B, num_classes]
        probs = torch.softmax(logits, dim=-1)  # [B, num_classes]
        confidence, _ = torch.max(probs, dim=-1, keepdim=True)  # [B, 1]

        return logits, probs, confidence

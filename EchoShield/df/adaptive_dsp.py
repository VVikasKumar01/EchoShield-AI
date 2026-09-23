from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class SubbandNLMS(nn.Module):
    """Subband Normalized Least Mean Squares (NLMS) Adaptive Residual Filter.

    Operates directly on the complex STFT bins (F = 481 bins) using reference microphone signals.
    Cancels coherent environmental noise (e.g. tank engine rumblings, 50/60 Hz generator hums,
    steady drone rotor acoustics) that leak past neural network filters.

    Adaptation step size mu(t) is AI-governed by the Noise Characterizer:
      - Accelerated during stationary noise (high p_stat)
      - Moderated during dynamic non-stationary transitions
      - Frozen (mu -> 0) during acoustic shock impulses to prevent filter divergence.
    """

    def __init__(
        self,
        num_freqs: int = 481,
        filter_order: int = 2,
        mu_init: float = 0.1,
        delta: float = 1e-4,
    ):
        """Initialize the subband NLMS filter.

        Args:
            num_freqs (int): Number of STFT frequency bins (typically 481 for 48kHz / 960 FFT).
            filter_order (int): Number of complex taps per subband (K >= 1).
            mu_init (float): Base adaptation step size.
            delta (float): Regularization constant to avoid division by zero.
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.filter_order = filter_order
        self.mu_init = mu_init
        self.delta = delta

    def forward(
        self,
        primary_spec: Tensor,
        ref_spec: Tensor,
        noise_probs: Optional[Tensor] = None,
        impulse_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Filter primary signal using reference microphone signal.

        Args:
            primary_spec (Tensor): Primary signal spectrum (real, imag) of shape [B, 1, T, F, 2]
                or [B, T, F, 2]. Represents AI-enhanced speech or primary mic.
            ref_spec (Tensor): Reference microphone spectrum of shape [B, 1, T, F, 2] or [B, T, F, 2].
            noise_probs (Tensor, optional): Probabilities [p_stat, p_non_stat, p_imp] of shape [B, 3]
                or [B, T, 3].
            impulse_mask (Tensor, optional): Impulse detection flags of shape [B, T, 1].

        Returns:
            error_spec (Tensor): Residual-filtered spectrum of shape matching primary_spec.
            noise_est_spec (Tensor): Estimated correlated noise spectrum subtracted from primary.
        """
        is_5d = primary_spec.dim() == 5
        if is_5d:
            d = primary_spec.squeeze(1)  # [B, T, F, 2]
            x = ref_spec.squeeze(1)  # [B, T, F, 2]
        else:
            d = primary_spec
            x = ref_spec

        b, t, f, _ = d.shape
        device = d.device

        # Real and imaginary components for subband matrix math (fully ONNX exportable)
        d_re, d_im = d[..., 0], d[..., 1]  # [B, T, F]
        x_re, x_im = x[..., 0], x[..., 1]  # [B, T, F]

        # Weights: [B, F, K] real and imag
        w_re = torch.zeros(b, f, self.filter_order, device=device)
        w_im = torch.zeros(b, f, self.filter_order, device=device)
        # Input buffer for reference history: [B, F, K] real and imag
        x_buf_re = torch.zeros(b, f, self.filter_order, device=device)
        x_buf_im = torch.zeros(b, f, self.filter_order, device=device)

        e_re = torch.zeros(b, t, f, device=device)
        e_im = torch.zeros(b, t, f, device=device)
        y_re = torch.zeros(b, t, f, device=device)
        y_im = torch.zeros(b, t, f, device=device)

        # Base step-size schedule per frame: [B, T, 1]
        if noise_probs is not None:
            if noise_probs.dim() == 2:
                # [B, 3] -> expand to [B, T, 3]
                probs = noise_probs.unsqueeze(1).expand(-1, t, -1)
            else:
                probs = noise_probs

            p_stat = probs[..., 0:1]  # [B, T, 1]
            p_non_stat = probs[..., 1:2]
            p_imp = probs[..., 2:3]

            # Fast step-size on stationary noise, slow on non-stationary
            mu_t = self.mu_init * (p_stat * 1.0 + p_non_stat * 0.3)
            # Freeze adaptation if impulse detected
            freeze = p_imp > 0.35
            if impulse_mask is not None:
                freeze = freeze | (impulse_mask > 0.5)
            mu_t = torch.where(freeze, torch.zeros_like(mu_t), mu_t)
        else:
            mu_t = torch.full((b, t, 1), self.mu_init, device=device)
            if impulse_mask is not None:
                mu_t = torch.where(impulse_mask > 0.5, torch.zeros_like(mu_t), mu_t)

        # Iterate over time frames (streaming subband adaptation)
        for n in range(t):
            # Update reference history buffer: roll and insert current frame at index 0
            x_buf_re = torch.roll(x_buf_re, shifts=1, dims=-1)
            x_buf_im = torch.roll(x_buf_im, shifts=1, dims=-1)
            x_buf_re[..., 0] = x_re[:, n, :]  # [B, F]
            x_buf_im[..., 0] = x_im[:, n, :]  # [B, F]

            # Estimated noise: y = w^H * x = sum(conj(w) * x, dim=-1)
            # conj(w) * x = (w_re - j*w_im) * (x_re + j*x_im) = (w_re*x_re + w_im*x_im) + j*(w_re*x_im - w_im*x_re)
            y_frame_re = torch.sum(w_re * x_buf_re + w_im * x_buf_im, dim=-1)  # [B, F]
            y_frame_im = torch.sum(w_re * x_buf_im - w_im * x_buf_re, dim=-1)  # [B, F]
            y_re[:, n, :] = y_frame_re
            y_im[:, n, :] = y_frame_im

            # Error: e = d - y
            e_frame_re = d_re[:, n, :] - y_frame_re  # [B, F]
            e_frame_im = d_im[:, n, :] - y_frame_im  # [B, F]
            e_re[:, n, :] = e_frame_re
            e_im[:, n, :] = e_frame_im

            # Reference power in each subband: ||x||^2 + delta
            x_power = torch.sum(x_buf_re ** 2 + x_buf_im ** 2, dim=-1, keepdim=True) + self.delta  # [B, F, 1]

            # Normalized tap update: w = w + (mu / (||x||^2 + delta)) * x * conj(e)
            # x * conj(e) = (x_re + j*x_im) * (e_re - j*e_im) = (x_re*e_re + x_im*e_im) + j*(x_im*e_re - x_re*e_im)
            scale = (mu_t[:, n, :].unsqueeze(1) / x_power)  # [B, F, 1]
            e_re_exp = e_frame_re.unsqueeze(-1)
            e_im_exp = e_frame_im.unsqueeze(-1)

            step_re = scale * (x_buf_re * e_re_exp + x_buf_im * e_im_exp)
            step_im = scale * (x_buf_im * e_re_exp - x_buf_re * e_im_exp)

            w_re = w_re + step_re
            w_im = w_im + step_im

        # Convert back to real/imag pair [B, T, F, 2]
        error_out = torch.stack([e_re, e_im], dim=-1)
        noise_est_out = torch.stack([y_re, y_im], dim=-1)

        if is_5d:
            error_out = error_out.unsqueeze(1)
            noise_est_out = noise_est_out.unsqueeze(1)

        return error_out, noise_est_out

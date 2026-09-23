from typing import Final, List, Optional, Tuple

import torch
from loguru import logger
from torch import Tensor, nn

import df.multiframe as MF
from df.adaptive_dsp import SubbandNLMS
from df.config import Csv, DfParams, config
from df.deepfilternet3 import DfDecoder, DfOutputReshapeMF, Encoder, ErbDecoder
from df.modules import (
    InstantaneousImpulseLimiter,
    Mask,
    erb_fb,
    get_device,
)
from df.noise_classifier import DefenseNoiseClassifier
from df.utils import as_complex
from libdf import DF

PI = 3.1415926535897932384626433


class ModelParams(DfParams):
    section = "deepfilternet"

    def __init__(self):
        super().__init__()
        # Backbone DFN3 parameters
        self.conv_lookahead: int = config(
            "CONV_LOOKAHEAD", cast=int, default=0, section=self.section
        )
        self.conv_ch: int = config("CONV_CH", cast=int, default=16, section=self.section)
        self.conv_depthwise: bool = config(
            "CONV_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.convt_depthwise: bool = config(
            "CONVT_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.conv_kernel: List[int] = config(
            "CONV_KERNEL", cast=Csv(int), default=(1, 3), section=self.section  # type: ignore
        )
        self.convt_kernel: List[int] = config(
            "CONVT_KERNEL", cast=Csv(int), default=(1, 3), section=self.section  # type: ignore
        )
        self.conv_kernel_inp: List[int] = config(
            "CONV_KERNEL_INP", cast=Csv(int), default=(3, 3), section=self.section  # type: ignore
        )
        self.emb_hidden_dim: int = config(
            "EMB_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.emb_num_layers: int = config(
            "EMB_NUM_LAYERS", cast=int, default=2, section=self.section
        )
        self.emb_gru_skip_enc: str = config(
            "EMB_GRU_SKIP_ENC", default="none", section=self.section
        )
        self.emb_gru_skip: str = config("EMB_GRU_SKIP", default="none", section=self.section)
        self.df_hidden_dim: int = config(
            "DF_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.df_gru_skip: str = config("DF_GRU_SKIP", default="none", section=self.section)
        self.df_pathway_kernel_size_t: int = config(
            "DF_PATHWAY_KERNEL_SIZE_T", cast=int, default=1, section=self.section
        )
        self.enc_concat: bool = config("ENC_CONCAT", cast=bool, default=False, section=self.section)
        self.df_num_layers: int = config("DF_NUM_LAYERS", cast=int, default=3, section=self.section)
        self.df_n_iter: int = config("DF_N_ITER", cast=int, default=1, section=self.section)
        self.lin_groups: int = config("LINEAR_GROUPS", cast=int, default=1, section=self.section)
        self.enc_lin_groups: int = config(
            "ENC_LINEAR_GROUPS", cast=int, default=16, section=self.section
        )
        self.mask_pf: bool = config("MASK_PF", cast=bool, default=False, section=self.section)
        self.pf_beta: float = config("PF_BETA", cast=float, default=0.02, section=self.section)
        self.lsnr_dropout: bool = config(
            "LSNR_DROPOUT", cast=bool, default=False, section=self.section
        )

        # Defense-specific extensions
        self.enable_noise_classifier: bool = config(
            "ENABLE_NOISE_CLASSIFIER", cast=bool, default=True, section=self.section
        )
        self.enable_impulse_limiter: bool = config(
            "ENABLE_IMPULSE_LIMITER", cast=bool, default=True, section=self.section
        )
        self.enable_adaptive_dsp: bool = config(
            "ENABLE_ADAPTIVE_DSP", cast=bool, default=True, section=self.section
        )
        self.adaptive_dsp_order: int = config(
            "ADAPTIVE_DSP_ORDER", cast=int, default=2, section=self.section
        )
        self.adaptive_dsp_mu: float = config(
            "ADAPTIVE_DSP_MU", cast=float, default=0.15, section=self.section
        )
        self.crest_thresh_db: float = config(
            "CREST_THRESH_DB", cast=float, default=12.0, section=self.section
        )
        self.energy_thresh_rel: float = config(
            "ENERGY_THRESH_REL", cast=float, default=8.0, section=self.section
        )
        self.max_atten_db: float = config(
            "MAX_ATTEN_DB", cast=float, default=24.0, section=self.section
        )


def init_model(df_state: Optional[DF] = None, run_df: bool = True, train_mask: bool = True):
    """Initialize the Defense-Enhanced DeepFilterNet model."""
    p = ModelParams()
    if df_state is None:
        df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb = erb_fb(df_state.erb_widths(), p.sr, inverse=False)
    erb_inverse = erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    model = DefenseDfNet(erb, erb_inverse, run_df, train_mask)
    return model.to(device=get_device())


class AdaptiveFusionController(nn.Module):
    """Dynamically blends AI-enhanced speech with Adaptive DSP residual output.

    Weighting policy:
      - Stationary noise (generators, vehicles): DSP residual subtraction is highly effective
        at removing static harmonics. alpha_dsp increases up to 0.7.
      - Dynamic / non-stationary (sirens, varying rotors): Neural Deep Filtering dominates.
        alpha_df approaches 1.0.
      - Impulsive shock (gunfire, artillery): DSP adaptation is frozen; impulse limiter and
        neural deep filtering suppress the shockwave without residual filter distortion.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        df_spec: Tensor,
        dsp_spec: Tensor,
        probs: Tensor,
        lsnr: Tensor,
        impulse_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Blend spectral predictions based on environmental regime and local SNR.

        Args:
            df_spec (Tensor): DeepFilterNet output [B, 1, T, F, 2].
            dsp_spec (Tensor): Subband NLMS residual output [B, 1, T, F, 2].
            probs (Tensor): Noise classifier probabilities [B, 3] or [B, T, 3].
            lsnr (Tensor): Local SNR estimate [B, T, 1].
            impulse_mask (Tensor): Impulse indicator [B, T, 1].

        Returns:
            fused_spec (Tensor): Blended enhanced speech spectrum [B, 1, T, F, 2].
            fusion_weights (Tensor): Alpha weights [B, T, 2] (alpha_df, alpha_dsp).
        """
        b, _, t, f, _ = df_spec.shape
        if probs.dim() == 2:
            p = probs.unsqueeze(1).expand(-1, t, -1)  # [B, T, 3]
        else:
            p = probs

        p_stat = p[..., 0:1]  # [B, T, 1]
        p_non_stat = p[..., 1:2]
        p_imp = p[..., 2:3]

        # In stationary noise, utilize DSP residual subtraction; in non-stationary, rely on AI
        alpha_dsp = torch.clamp(p_stat * 0.7 + p_non_stat * 0.2, 0.0, 0.8)

        # In impulse shocks, immediately freeze/suppress DSP output
        is_impulse = (p_imp > 0.3) | (impulse_mask > 0.5)
        alpha_dsp = torch.where(is_impulse, torch.zeros_like(alpha_dsp), alpha_dsp)

        alpha_df = 1.0 - (0.5 * alpha_dsp)

        # Reshape weights for broadcasting: [B, 1, T, 1, 1]
        w_df = alpha_df.unsqueeze(1).unsqueeze(-1)
        w_dsp = alpha_dsp.unsqueeze(1).unsqueeze(-1)

        fused = w_df * df_spec + w_dsp * dsp_spec
        weights = torch.cat([alpha_df, alpha_dsp], dim=-1)  # [B, T, 2]

        return fused, weights


class DefenseDfNet(nn.Module):
    """Defense-Enhanced DeepFilterNet.

    Combines:
      - Block 1 & 2: Dual-channel STFT & ERB representation (Primary + Reference mic)
      - Block 3: DefenseNoiseClassifier (Stationary, Non-stationary, Impulsive)
      - Block 4: InstantaneousImpulseLimiter + DFN3 Neural Core
      - Block 5: Subband NLMS Adaptive Residual Filter
      - Block 6: Adaptive Fusion Controller
      - Block 7: ISTFT Synthesis-ready output
    """

    freq_bins: Final[int]
    emb_dim: Final[int]
    erb_bins: Final[int]
    nb_df: Final[int]
    df_order: Final[int]
    df_lookahead: Final[int]

    def __init__(
        self,
        erb_fb: Tensor,
        erb_inv_fb: Tensor,
        run_df: bool = True,
        train_mask: bool = True,
    ):
        super().__init__()
        p = ModelParams()
        self.freq_bins = p.fft_size // 2 + 1
        self.emb_dim = p.emb_hidden_dim
        self.erb_bins = p.nb_erb
        self.nb_df = p.nb_df
        self.df_order = p.df_order
        self.df_lookahead = p.df_lookahead

        if p.conv_lookahead > 0:
            self.pad_feat = nn.ConstantPad2d((0, 0, -p.conv_lookahead, p.conv_lookahead), 0.0)
        else:
            self.pad_feat = nn.Identity()

        if p.df_lookahead > 0:
            self.pad_spec = nn.ConstantPad3d((0, 0, 0, 0, -p.df_lookahead, p.df_lookahead), 0.0)
        else:
            self.pad_spec = nn.Identity()

        self.register_buffer("erb_fb", erb_fb)
        self.erb_inv_fb = erb_inv_fb

        # Block 4: Impulse protection layer
        self.enable_impulse_limiter = p.enable_impulse_limiter
        if self.enable_impulse_limiter:
            self.impulse_limiter = InstantaneousImpulseLimiter(
                crest_thresh_db=p.crest_thresh_db,
                energy_thresh_rel=p.energy_thresh_rel,
                max_atten_db=p.max_atten_db,
            )
        else:
            self.impulse_limiter = None

        # Block 3: Noise characterization
        self.enable_noise_classifier = p.enable_noise_classifier
        if self.enable_noise_classifier:
            self.noise_classifier = DefenseNoiseClassifier(in_channels=p.nb_erb)
        else:
            self.noise_classifier = None

        # Core DFN3 Backbone
        self.enc = Encoder()
        self.erb_dec = ErbDecoder()
        self.mask = Mask(erb_inv_fb)
        self.post_filter = p.mask_pf
        self.post_filter_beta = p.pf_beta

        self.df_op = MF.DF(num_freqs=p.nb_df, frame_size=p.df_order, lookahead=self.df_lookahead)
        self.df_dec = DfDecoder()
        self.df_out_transform = DfOutputReshapeMF(self.df_order, p.nb_df)

        self.run_erb = p.nb_df + 1 < self.freq_bins
        self.run_df = run_df
        self.train_mask = train_mask
        self.lsnr_dropout = p.lsnr_dropout

        # Block 5: Adaptive Residual Filter
        self.enable_adaptive_dsp = p.enable_adaptive_dsp
        if self.enable_adaptive_dsp:
            self.adaptive_dsp = SubbandNLMS(
                num_freqs=self.freq_bins,
                filter_order=p.adaptive_dsp_order,
                mu_init=p.adaptive_dsp_mu,
            )
        else:
            self.adaptive_dsp = None

        # Block 6: Adaptive Fusion Controller
        self.fusion_controller = AdaptiveFusionController()

    def forward(
        self,
        spec: Tensor,
        feat_erb: Tensor,
        feat_spec: Tensor,
        ref_spec: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward pass of Defense-Enhanced DeepFilterNet.

        Args:
            spec (Tensor): Primary microphone spectrum of shape [B, 1, T, F, 2].
            feat_erb (Tensor): ERB energy features of shape [B, 1, T, E].
            feat_spec (Tensor): Complex STFT features of shape [B, 1, T, F', 2].
            ref_spec (Tensor, optional): Optional reference microphone spectrum [B, 1, T, F, 2].
                If None, a synthesized ambient reference or self-residual is utilized.

        Returns:
            spec_fused (Tensor): Final enhanced spectrum of shape [B, 1, T, F, 2].
            mask (Tensor): ERB mask estimate of shape [B, 1, T, E].
            lsnr (Tensor): Local SNR estimate of shape [B, T, 1].
            df_coefs (Tensor): Complex DF filter coefficients.
            noise_probs (Tensor): Environmental regime distribution [B, 3].
            impulse_mask (Tensor): Impulsive shock indicator [B, T, 1].
            fusion_weights (Tensor): AI vs DSP fusion weights [B, T, 2].
        """
        # Block 4: Check and apply instantaneous impulse shock limiting
        if self.impulse_limiter is not None:
            spec_in, feat_erb_in, impulse_mask = self.impulse_limiter(spec, feat_erb)
        else:
            spec_in, feat_erb_in = spec, feat_erb
            impulse_mask = torch.zeros(spec.shape[0], spec.shape[2], 1, device=spec.device)

        # Block 3: Classify noise environment
        if self.noise_classifier is not None:
            _, noise_probs, _ = self.noise_classifier(feat_erb)
        else:
            # Default to neutral distribution
            noise_probs = torch.tensor([[0.34, 0.33, 0.33]], device=spec.device).repeat(spec.shape[0], 1)

        # Core DFN3 Encoder
        feat_spec_proc = feat_spec.squeeze(1).permute(0, 3, 1, 2)
        feat_erb_proc = self.pad_feat(feat_erb_in)
        feat_spec_proc = self.pad_feat(feat_spec_proc)
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb_proc, feat_spec_proc)

        # ERB Mask Decoder
        if self.run_erb:
            m = self.erb_dec(emb, e3, e2, e1, e0)
            spec_m = self.mask(spec_in, m)
        else:
            m = torch.zeros((), device=spec.device)
            spec_m = spec_in

        # DF Coefficient Decoder
        if self.run_df:
            df_coefs = self.df_dec(emb, c0)
            df_coefs = self.df_out_transform(df_coefs)
            spec_df = self.df_op(spec_in.clone(), df_coefs)
            spec_df[..., self.nb_df :, :] = spec_m[..., self.nb_df :, :]
        else:
            df_coefs = torch.zeros((), device=spec.device)
            spec_df = spec_m

        # Valin post-filter
        if self.post_filter:
            beta = self.post_filter_beta
            eps = 1e-12
            mag_df = torch.sqrt(spec_df[..., 0] ** 2 + spec_df[..., 1] ** 2 + eps)
            mag_in = torch.sqrt(spec_in[..., 0] ** 2 + spec_in[..., 1] ** 2 + eps)
            mask_val = (mag_df / (mag_in + eps)).clamp(eps, 1)
            mask_sin = mask_val * torch.sin(PI * mask_val / 2).clamp_min(eps)
            pf = (1 + beta) / (1 + beta * mask_val.div(mask_sin).pow(2))
            spec_df = spec_df * pf.unsqueeze(-1)

        # Block 5: Adaptive Residual Filter (Subband NLMS)
        if self.enable_adaptive_dsp:
            # If no physical reference mic provided, synthesize one from the noisy-speech difference
            if ref_spec is None:
                effective_ref = spec - spec_df.detach()
            else:
                effective_ref = ref_spec

            spec_dsp, _ = self.adaptive_dsp(
                primary_spec=spec_df,
                ref_spec=effective_ref,
                noise_probs=noise_probs,
                impulse_mask=impulse_mask,
            )
        else:
            spec_dsp = spec_df

        # Block 6: Adaptive Fusion Controller
        spec_fused, fusion_weights = self.fusion_controller(
            df_spec=spec_df,
            dsp_spec=spec_dsp,
            probs=noise_probs,
            lsnr=lsnr,
            impulse_mask=impulse_mask,
        )

        return spec_fused, m, lsnr, df_coefs, noise_probs, impulse_mask, fusion_weights

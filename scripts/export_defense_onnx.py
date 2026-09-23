"""
export_defense_onnx.py
=======================
Phase 4 of the DeepFilterNet Defence Upgrade Experimental Roadmap.

Exports DefenseDfNet subgraphs to ONNX for:
  – Embedded C-ABI deployment (via ONNX Runtime)
  – Tract inference engine (Rust-based, MCU-capable)
  – TensorFlow Lite conversion pipeline (via onnx-tf)

Export targets
--------------
  1. full_model.onnx          – Complete DefenseDfNet (encoder + decoders +
                                 noise classifier + fusion controller).
                                 For server / edge GPU / high-end ARM deployments.

  2. neural_core.onnx         – DFN3 backbone only (enc + erb_dec + df_dec).
                                 Smallest footprint; for MCU targets.

  3. noise_classifier.onnx    – Stand-alone 3-class noise classifier.
                                 Can run asynchronously on a second thread.

Usage
-----
  python scripts/export_defense_onnx.py \\
      --ckpt   models/DeepFilterNet_defense_init.ckpt \\
      --out_dir exports/onnx

  # Or export from random weights (for architecture validation):
  python scripts/export_defense_onnx.py --random_weights --out_dir exports/onnx
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "DeepFilterNet"))

SR = 48_000
N_FFT_BINS = 481   # (960 // 2) + 1
ERB_BANDS  = 32
DF_BINS    = 96

# Export batch size and temporal length (adjust for your deployment target)
EXPORT_B  = 1
EXPORT_T  = 1   # single frame (streaming / causal)


# ===========================================================================
# Wrapper modules for clean ONNX export
# (avoids tuple-output issues in older opset versions)
# ===========================================================================

class DfOpONNXWrapper(nn.Module):
    """ONNX-exportable real-valued Deep Filtering operator."""
    def __init__(self, df_bins: int, df_order: int, df_lookahead: int = 0):
        super().__init__()
        self.df_bins = df_bins
        self.df_order = df_order
        self.df_lookahead = df_lookahead

    def forward(self, spec: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        # spec: [B, 1, T, F, 2], coefs: [B, O, T, F, 2]
        b, c, t, f, _ = spec.shape
        spec_sub = spec[:, 0, :, :self.df_bins, :]  # [B, T, F_df, 2]
        c_perm = coefs.permute(0, 2, 1, 3, 4)       # [B, T, O, F_df, 2]

        out_re = torch.zeros(b, t, self.df_bins, device=spec.device)
        out_im = torch.zeros(b, t, self.df_bins, device=spec.device)

        for i in range(self.df_order):
            c_i = c_perm[:, :, i, :self.df_bins, :]
            out_re = out_re + (spec_sub[..., 0] * c_i[..., 0] - spec_sub[..., 1] * c_i[..., 1])
            out_im = out_im + (spec_sub[..., 0] * c_i[..., 1] + spec_sub[..., 1] * c_i[..., 0])

        out = torch.stack([out_re, out_im], dim=-1).unsqueeze(1)  # [B, 1, T, F_df, 2]
        spec_out = spec.clone()
        spec_out[..., :self.df_bins, :] = out
        return spec_out


class FullModelWrapper(nn.Module):
    """
    Wraps DefenseDfNet and exposes a single (spec_out,) output for ONNX.
    The full model takes three inputs and returns the fused spectrum.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.df_op = DfOpONNXWrapper(
            df_bins=self.model.nb_df,
            df_order=self.model.df_order,
            df_lookahead=self.model.df_lookahead,
        )

    def forward(self,
                spec:      torch.Tensor,   # [B, 1, T, 481, 2]
                feat_erb:  torch.Tensor,   # [B, 1, T, 32]
                feat_spec: torch.Tensor,   # [B, 1, T, 96, 2]
                ) -> torch.Tensor:
        out = self.model(spec, feat_erb, feat_spec)
        return out[0]   # fused spectrum [B, 1, T, 481, 2]


class NeuralCoreWrapper(nn.Module):
    """
    Wraps only the ERB encoder + ERB decoder + DF decoder (DFN3 backbone).
    Useful for the smallest embedded footprint.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.enc     = model.enc
        self.erb_dec = model.erb_dec
        self.df_dec  = model.df_dec
        self.pad_feat = getattr(model, "pad_feat", nn.Identity())

    def forward(self,
                feat_erb:  torch.Tensor,   # [B, 1, T, 32]
                feat_spec: torch.Tensor,   # [B, 1, T, 96, 2]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_spec_proc = feat_spec.squeeze(1).permute(0, 3, 1, 2)  # [B, 2, T, 96]
        feat_erb_proc = self.pad_feat(feat_erb)
        feat_spec_proc = self.pad_feat(feat_spec_proc)
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb_proc, feat_spec_proc)
        # ERB mask
        erb_mask = self.erb_dec(emb, e3, e2, e1, e0)
        # DF coefficients
        df_coefs = self.df_dec(emb, c0)
        return erb_mask, df_coefs


class NoiseClassifierWrapper(nn.Module):
    """Stand-alone noise classifier export."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.classifier = model.noise_classifier

    def forward(self, feat_erb: torch.Tensor) -> torch.Tensor:
        # feat_erb: [B, 1, T, 32]
        _, probs, _ = self.classifier(feat_erb)
        return probs   # [B, 3]


# ===========================================================================
# ONNX export helper
# ===========================================================================

def export_onnx(module: nn.Module,
                example_inputs: tuple,
                input_names: list[str],
                output_names: list[str],
                out_path: str,
                opset: int = 17,
                dynamic_axes: dict | None = None) -> None:
    """Export a module to ONNX and optionally validate with onnxruntime."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    module.eval()
    with torch.no_grad():
        try:
            torch.onnx.export(
                module,
                example_inputs,
                out_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes or {},
                verbose=False,
                dynamo=False,
            )
        except TypeError:
            # Older PyTorch versions do not have dynamo kwarg
            torch.onnx.export(
                module,
                example_inputs,
                out_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes or {},
                verbose=False,
            )

    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"  [OK]  Saved: {out_path}  ({size_mb:.1f} MB)")

    # Validate with onnxruntime if available
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(out_path,
                                    providers=["CPUExecutionProvider"])
        feed = {
            sess.get_inputs()[i].name: example_inputs[i].numpy()
            for i in range(len(example_inputs))
        }
        outs = sess.run(None, feed)
        print(f"     ORT validation OK — output shape: {outs[0].shape}")
    except ImportError:
        print("     (onnxruntime not installed — skipping ORT validation)")
    except Exception as e:
        print(f"     [WARNING] ORT validation failed: {e}")

    # Validate with onnx checker if available
    try:
        import onnx
        model_proto = onnx.load(out_path)
        onnx.checker.check_model(model_proto)
        print("     ONNX checker: PASS")
    except ImportError:
        print("     (onnx package not installed — skipping checker)")
    except Exception as e:
        print(f"     [WARNING] ONNX checker: {e}")


# ===========================================================================
# Main export pipeline
# ===========================================================================

def export_all(ckpt_path: str | None, out_dir: str,
               random_weights: bool = False, opset: int = 17) -> None:
    print(f"\n{'='*64}")
    print(f"  DefenseDfNet -> ONNX Export Pipeline")
    print(f"  Opset: {opset} | Output: {os.path.abspath(out_dir)}")
    print(f"{'='*64}\n")

    # -- Load model ---------------------------------------------------------
    from df.config import config
    config.use_defaults(allow_reload=True)
    from df.deepfilternet_defense import init_model

    model = init_model()

    if not random_weights and ckpt_path:
        if not os.path.isfile(ckpt_path):
            print(f"  [WARNING] Checkpoint not found: {ckpt_path}")
            print("  Using random weights (pass --random_weights to silence this)")
        else:
            print(f"[Step 1] Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model", ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"  Loaded - missing: {len(missing)}, unexpected: {len(unexpected)}")
    else:
        print("[Step 1] Using random weights (architecture validation mode)")

    model.eval()

    # -- Example input tensors ----------------------------------------------
    B, T = EXPORT_B, EXPORT_T
    spec      = torch.randn(B, 1, T, N_FFT_BINS, 2)
    feat_erb  = torch.randn(B, 1, T, ERB_BANDS)
    feat_spec = torch.randn(B, 1, T, DF_BINS, 2)

    # Dynamic axes: allow batch and time to vary at inference
    dynamic_full = {
        "spec":      {0: "batch", 2: "time"},
        "feat_erb":  {0: "batch", 2: "time"},
        "feat_spec": {0: "batch", 2: "time"},
        "spec_out":  {0: "batch", 2: "time"},
    }

    # -- Export 1: Full model -----------------------------------------------
    print("\n[Export 1] Full DefenseDfNet …")
    try:
        full_wrapper = FullModelWrapper(model)
        export_onnx(
            full_wrapper,
            (spec, feat_erb, feat_spec),
            input_names=["spec", "feat_erb", "feat_spec"],
            output_names=["spec_out"],
            out_path=os.path.join(out_dir, "full_model.onnx"),
            opset=opset,
            dynamic_axes=dynamic_full,
        )
    except Exception as e:
        print(f"  [ERROR] Full model export failed: {e}")

    # -- Export 2: Neural core only -----------------------------------------
    print("\n[Export 2] Neural core (DFN3 backbone only) …")
    try:
        core_wrapper = NeuralCoreWrapper(model)
        core_wrapper.eval()
        export_onnx(
            core_wrapper,
            (feat_erb, feat_spec),
            input_names=["feat_erb", "feat_spec"],
            output_names=["erb_mask", "df_coefs"],
            out_path=os.path.join(out_dir, "neural_core.onnx"),
            opset=opset,
            dynamic_axes={
                "feat_erb":  {0: "batch", 2: "time"},
                "feat_spec": {0: "batch", 2: "time"},
                "erb_mask":  {0: "batch", 2: "time"},
                "df_coefs":  {0: "batch", 2: "time"},
            },
        )
    except Exception as e:
        print(f"  [ERROR] Neural core export failed: {e}")
        print("  (This may require backbone attribute names to be public — "
              "see ModelParams.enable_* flags.)")

    # -- Export 3: Noise classifier -----------------------------------------
    print("\n[Export 3] Noise classifier …")
    try:
        cls_wrapper = NoiseClassifierWrapper(model)
        cls_wrapper.eval()
        export_onnx(
            cls_wrapper,
            (feat_erb,),
            input_names=["feat_erb"],
            output_names=["class_probs"],
            out_path=os.path.join(out_dir, "noise_classifier.onnx"),
            opset=opset,
            dynamic_axes={
                "feat_erb":    {0: "batch", 2: "time"},
                "class_probs": {0: "batch"},
            },
        )
    except Exception as e:
        print(f"  [ERROR] Classifier export failed: {e}")

    # -- Manifest -----------------------------------------------------------
    import json
    manifest = {
        "export_timestamp": str(__import__("datetime").datetime.now()),
        "opset": opset,
        "sr_hz": SR,
        "n_fft_bins": N_FFT_BINS,
        "erb_bands": ERB_BANDS,
        "df_bins": DF_BINS,
        "exports": {
            "full_model":        "full_model.onnx",
            "neural_core":       "neural_core.onnx",
            "noise_classifier":  "noise_classifier.onnx",
        },
        "source_checkpoint": ckpt_path,
    }
    manifest_path = os.path.join(out_dir, "export_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n  Manifest: {manifest_path}")

    print(f"\n{'='*64}")
    print(f"  ONNX export pipeline COMPLETE")
    print(f"{'='*64}\n")


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DefenseDfNet subgraphs to ONNX."
    )
    parser.add_argument(
        "--ckpt", default="models/DeepFilterNet_defense_init.ckpt",
        help="Path to DefenseDfNet checkpoint (from init_defense_checkpoint.py)"
    )
    parser.add_argument(
        "--out_dir", default="exports/onnx",
        help="Output directory for .onnx files"
    )
    parser.add_argument(
        "--random_weights", action="store_true",
        help="Export with random weights (architecture validation only)"
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17, minimum: 13)"
    )
    args = parser.parse_args()
    export_all(args.ckpt, args.out_dir, args.random_weights, args.opset)


if __name__ == "__main__":
    main()

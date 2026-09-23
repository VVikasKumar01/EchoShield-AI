"""
init_defense_checkpoint.py
===========================
Phase 2 of the DeepFilterNet Defence Upgrade Experimental Roadmap.

Transfers backbone weights from the official pretrained DeepFilterNet3 checkpoint
into the DefenseDfNet architecture.  New defence-specific modules are initialised
with identity/zero-distortion defaults.

Weight-mapping strategy
-----------------------
  DeepFilterNet3 state keys   → DefenseDfNet state keys
  ──────────────────────────    ──────────────────────────
  enc.*                       → enc.*          (direct copy)
  erb_dec.*                   → erb_dec.*      (direct copy)
  df_dec.*                    → df_dec.*       (direct copy)
  mask.*                      → mask.*         (direct copy)
  df_op.*                     → df_op.*        (direct copy)

New modules (zero-distortion initialisation):
  noise_classifier.*          → tiny CNN, weights reset to 0 (bias=const)
  impulse_limiter.*           → no learnable params (pure DSP)
  adaptive_dsp.*              → taps set to 0 (filter starts transparent)
  fusion_controller.*         → output gate bias set to favour alpha_DF=1.0

Usage
-----
  python scripts/init_defense_checkpoint.py \\
      --dfn3_zip   models/DeepFilterNet3.zip \\
      --out_ckpt   models/DeepFilterNet_defense_init.ckpt
"""

import argparse
import io
import os
import sys
import zipfile

import torch

# ---------------------------------------------------------------------------
# Make sure DeepFilterNet package is importable
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "DeepFilterNet"))

from df.config import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_ckpt_in_zip(zf: zipfile.ZipFile) -> str | None:
    """Return the path of the 'best' checkpoint inside the zip archive."""
    candidates = [n for n in zf.namelist() if n.endswith(".ckpt.best")]
    if not candidates:
        candidates = [n for n in zf.namelist() if n.endswith(".ckpt")]
    if not candidates:
        return None
    # Prefer highest epoch number
    def epoch_key(name):
        parts = os.path.basename(name).replace(".ckpt.best", "").replace(".ckpt", "").split("_")
        for p in reversed(parts):
            try:
                return int(p)
            except ValueError:
                pass
        return 0
    return sorted(candidates, key=epoch_key)[-1]


def load_dfn3_state(dfn3_zip: str) -> dict:
    """
    Extract and load the DeepFilterNet3 checkpoint from a .zip file.
    Returns the 'model' sub-dict if present, else the raw state_dict.
    """
    if not os.path.isfile(dfn3_zip):
        raise FileNotFoundError(f"DFN3 zip not found: {dfn3_zip}")

    print(f"  Opening archive: {dfn3_zip}")
    with zipfile.ZipFile(dfn3_zip, "r") as zf:
        ckpt_name = find_ckpt_in_zip(zf)
        if ckpt_name is None:
            raise FileNotFoundError("No .ckpt file found inside the zip archive.")
        print(f"  Found checkpoint: {ckpt_name}")
        with zf.open(ckpt_name) as fh:
            buf = io.BytesIO(fh.read())
            raw = torch.load(buf, map_location="cpu", weights_only=True)

    # The checkpoint may be wrapped in {"model": state_dict, ...}
    if isinstance(raw, dict):
        if "model" in raw:
            return raw["model"]
        if "state_dict" in raw:
            return raw["state_dict"]
    return raw


def remap_keys(src: dict) -> dict:
    """
    Remap DFN3 state-dict keys to DefenseDfNet keys.
    Both architectures share the same backbone key names, so no renaming
    is needed for backbone params.  This function is kept as a hook for
    future remapping if key prefixes diverge.
    """
    remapped = {}
    for k, v in src.items():
        remapped[k] = v
    return remapped


def describe_coverage(dfn3_keys: set, defense_keys: set) -> None:
    """Print a summary of key coverage."""
    matched  = dfn3_keys & defense_keys
    only_dfn3  = dfn3_keys - defense_keys
    only_def   = defense_keys - dfn3_keys

    print(f"\n  Key coverage summary")
    print(f"  {'─'*50}")
    print(f"  Matched (backbone transferred) : {len(matched):>5}")
    print(f"  Only in DFN3 (ignored)         : {len(only_dfn3):>5}")
    print(f"  Only in DefenseDfNet (new)     : {len(only_def):>5}")

    if only_dfn3:
        print("\n  [WARNING] DFN3 keys NOT transferred (may indicate mismatch):")
        for k in sorted(only_dfn3)[:15]:
            print(f"    - {k}")
        if len(only_dfn3) > 15:
            print(f"    ... and {len(only_dfn3) - 15} more")

    if only_def:
        print("\n  [INFO] New DefenseDfNet keys (initialised fresh):")
        for k in sorted(only_def)[:15]:
            print(f"    + {k}")
        if len(only_def) > 15:
            print(f"    ... and {len(only_def) - 15} more")


def zero_distortion_init(model: torch.nn.Module) -> None:
    """
    Reset new defence-specific modules to zero-distortion defaults:
      - noise_classifier  : zero weights, uniform bias (avoids early saturation)
      - adaptive_dsp      : complex taps set to 0 (transparent filter)
      - fusion_controller : gate bias strongly favours DFN3 output (alpha_DF ≈ 1)
    """
    for name, module in model.named_modules():
        # Adaptive DSP: zero taps
        if "adaptive_dsp" in name:
            for param in module.parameters():
                torch.nn.init.zeros_(param)

        # Noise classifier: He init for conv, zero for linear
        if "noise_classifier" in name:
            if isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.zeros_(module.weight)
                if module.bias is not None:
                    # Uniform logits → equal prior across classes
                    torch.nn.init.constant_(module.bias, 1.0 / 3.0)

        # Fusion controller: bias gate toward alpha_DF = 1.0
        if "fusion_controller" in name and isinstance(module, torch.nn.Linear):
            torch.nn.init.zeros_(module.weight)
            if module.bias is not None:
                # Two outputs: [alpha_DF_logit, alpha_DSP_logit]
                # Set so sigmoid gives (1.0, 0.0)
                torch.nn.init.constant_(module.bias, 0.0)
                with torch.no_grad():
                    if module.bias.shape[0] >= 2:
                        module.bias[0] = 6.0   # alpha_DF  → sigmoid → ~1.0
                        module.bias[1] = -6.0  # alpha_DSP → sigmoid → ~0.0


# ===========================================================================
# Main transfer function
# ===========================================================================

def transfer_weights(dfn3_zip: str, out_ckpt: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_ckpt)), exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  DeepFilterNet3 → DefenseDfNet  Weight Transfer")
    print(f"{'='*64}\n")

    # ── Step 1: Load DFN3 checkpoint ──────────────────────────────────────
    print("[Step 1] Loading DFN3 checkpoint …")
    dfn3_state = load_dfn3_state(dfn3_zip)
    print(f"  DFN3 state-dict: {len(dfn3_state)} keys")

    # ── Step 2: Build DefenseDfNet skeleton ───────────────────────────────
    print("\n[Step 2] Instantiating DefenseDfNet ...")
    config.use_defaults(allow_reload=True)
    from df.deepfilternet_defense import init_model
    model = init_model()
    model.eval()

    defense_state = model.state_dict()
    print(f"  DefenseDfNet state-dict: {len(defense_state)} keys")

    # ── Step 3: Remap and transfer ─────────────────────────────────────────
    print("\n[Step 3] Remapping keys …")
    remapped = remap_keys(dfn3_state)
    dfn3_keys   = set(remapped.keys())
    defense_keys = set(defense_state.keys())

    describe_coverage(dfn3_keys, defense_keys)

    # Build new state dict: start from defense defaults, overwrite with DFN3
    new_state = defense_state.copy()
    n_transferred = 0
    n_skipped_shape = 0

    for k, v in remapped.items():
        if k in new_state:
            if new_state[k].shape == v.shape:
                new_state[k] = v
                n_transferred += 1
            else:
                print(f"  [SHAPE MISMATCH] {k}: DFN3={tuple(v.shape)} vs "
                      f"Defence={tuple(new_state[k].shape)}  — skipped")
                n_skipped_shape += 1

    print(f"\n  Transferred : {n_transferred} tensors")
    print(f"  Skipped (shape mismatch): {n_skipped_shape}")

    # ── Step 4: Load into model and apply zero-distortion init ────────────
    print("\n[Step 4] Loading merged state dict …")
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}):")
        for k in missing[:10]:
            print(f"    - {k}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}):")
        for k in unexpected[:10]:
            print(f"    + {k}")

    print("\n[Step 5] Applying zero-distortion defaults to new modules …")
    zero_distortion_init(model)

    # ── Step 5: Verify forward pass ───────────────────────────────────────
    print("\n[Step 6] Verifying forward pass …")
    with torch.no_grad():
        B, T = 1, 4
        n_fft, erb_bands, df_bins = 481, 32, 96
        spec     = torch.randn(B, 1, T, n_fft, 2)
        feat_erb = torch.randn(B, 1, T, erb_bands)
        feat_spec = torch.randn(B, 1, T, df_bins, 2)
        try:
            out = model(spec, feat_erb, feat_spec)
            print(f"  Forward pass OK — output spec shape: {out[0].shape}")
        except Exception as e:
            print(f"  [WARNING] Forward pass raised: {e}")
            print("  (Weight transfer still saved; model may need fine-tuning.)")

    # ── Step 6: Save initialised checkpoint ───────────────────────────────
    print(f"\n[Step 7] Saving initialised checkpoint …")
    save_dict = {
        "model":       model.state_dict(),
        "dfn3_source": dfn3_zip,
        "n_transferred": n_transferred,
        "note": (
            "DefenseDfNet initialised from DFN3 backbone. "
            "noise_classifier / adaptive_dsp / fusion_controller "
            "use zero-distortion defaults."
        ),
    }
    torch.save(save_dict, out_ckpt)
    size_mb = os.path.getsize(out_ckpt) / (1024 ** 2)
    print(f"  Saved: {out_ckpt}  ({size_mb:.1f} MB)")

    print(f"\n{'='*64}")
    print(f"  Weight transfer COMPLETE")
    print(f"  {n_transferred}/{len(dfn3_keys)} DFN3 backbone params transferred")
    print(f"{'='*64}\n")


# ===========================================================================
# CLI entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer DFN3 backbone weights into DefenseDfNet."
    )
    parser.add_argument(
        "--dfn3_zip", default="models/DeepFilterNet3.zip",
        help="Path to the pretrained DeepFilterNet3 .zip archive"
    )
    parser.add_argument(
        "--out_ckpt", default="models/DeepFilterNet_defense_init.ckpt",
        help="Output path for the initialised DefenseDfNet checkpoint"
    )
    args = parser.parse_args()
    transfer_weights(args.dfn3_zip, args.out_ckpt)


if __name__ == "__main__":
    main()

"""
generate_defense_dataset.py
============================
Phase 1 of the DeepFilterNet Defence Upgrade Experimental Roadmap.

Generates a synthetic military-noise test/train dataset covering three classes:
  Class 0 – Stationary   : field generator 50/60 Hz, tank diesel idle, steady drone hum
  Class 1 – Non-Stationary: MI-17/Blackhawk BPF sweep, emergency siren Doppler, RPM ramp
  Class 2 – Impulsive     : AK-47/rifle shots, machine-gun bursts, mortar blasts

Each noise track is mixed with the reference clean speech at five calibrated SNR
levels:  [-10, -5, 0, +5, +10] dB.

Outputs
-------
  <out_dir>/
    noisy/       – noisy mixture .wav files
    clean/       – aligned clean reference .wav files
    metadata.csv – per-file labels (class_id, snr, impulse_timestamps)
    stats.json   – dataset summary statistics

Usage
-----
  python scripts/generate_defense_dataset.py --out_dir data/defense_dataset
"""

import argparse
import csv
import json
import math
import os
import random
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Try to import soundfile; fall back to a wav writer using wave + array
# ---------------------------------------------------------------------------
try:
    import soundfile as sf
    _HAVE_SF = True
except ImportError:
    import wave
    _HAVE_SF = False

SR = 48_000          # Hz – matches DeepFilterNet 48 kHz pipeline
DURATION = 4.0       # seconds per clip
HOP = 480            # samples (10 ms)
N_SAMPLES = int(SR * DURATION)
SNR_GRID = [-10, -5, 0, 5, 10]   # dB


# ===========================================================================
# Helper utilities
# ===========================================================================

def rng_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def db_to_lin(db: float) -> float:
    return 10 ** (db / 20.0)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise so speech / noise power ratio equals snr_db, then add."""
    speech_rms = rms(speech)
    noise_rms  = rms(noise)
    target_noise_rms = speech_rms / db_to_lin(snr_db)
    scale = target_noise_rms / noise_rms
    return speech + scale * noise


def save_wav(path: str, data: np.ndarray, sr: int = SR) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data_clipped = np.clip(data, -1.0, 1.0).astype(np.float32)
    if _HAVE_SF:
        sf.write(path, data_clipped, sr, subtype="PCM_16")
    else:
        import wave, struct
        pcm = (data_clipped * 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())


# ===========================================================================
# Synthetic speech – frequency-modulated vowel cascade (no corpus needed)
# ===========================================================================

def gen_clean_speech(n: int, sr: int, gen: np.random.Generator) -> np.ndarray:
    """
    Synthesise a plausible speech-like waveform (voiced + unvoiced segments).
    Uses a sawtooth source with time-varying F0 filtered by formant resonators.
    """
    t = np.arange(n) / sr
    # Slowly varying F0 between 100-250 Hz
    f0 = 150 + 50 * np.sin(2 * math.pi * 0.5 * t)

    # Sawtooth (glottal source approximation)
    phase = np.cumsum(f0 / sr) % 1.0
    source = 2 * phase - 1   # sawtooth in [-1, 1]

    # Formant resonators: 3 second-order IIR bandpass filters
    try:
        from scipy.signal import butter, lfilter

        def formant(x, fc, bw):
            b, a = butter(2, [max(fc - bw/2, 20) / (sr/2),
                              min(fc + bw/2, sr/2 - 1) / (sr/2)], btype="band")
            return lfilter(b, a, x)

        sig = (formant(source, 800,  200)
             + formant(source, 1800, 300) * 0.5
             + formant(source, 2800, 400) * 0.25)
    except (ImportError, Exception):
        sig = source   # scipy not available: raw sawtooth fallback

    # Add unvoiced segments (uniform noise bursts)
    voiced_mask = (np.sin(2 * math.pi * 2.5 * t) > -0.3).astype(float)
    noise_burst = gen.uniform(-0.08, 0.08, n)
    sig = sig * voiced_mask + noise_burst * (1 - voiced_mask)

    sig /= (rms(sig) + 1e-8) * 10   # normalise to ~0.1 RMS
    return sig.astype(np.float32)


# ===========================================================================
# Noise generators – Class 0: Stationary
# ===========================================================================

def gen_field_generator(n: int, sr: int, gen: np.random.Generator,
                        freq_hz: float = 50.0) -> np.ndarray:
    """50/60 Hz generator hum with odd harmonics (1st, 3rd, 5th, 7th)."""
    t = np.arange(n) / sr
    sig = np.zeros(n)
    harmonics = [1, 3, 5, 7]
    amps = [1.0, 0.4, 0.2, 0.1]
    for h, a in zip(harmonics, amps):
        sig += a * np.sin(2 * math.pi * freq_hz * h * t
                          + gen.uniform(0, 2 * math.pi))
    # Low-frequency rumble
    sig += 0.3 * np.sin(2 * math.pi * 12 * t)
    sig /= rms(sig) + 1e-8
    return sig.astype(np.float32)


def gen_tank_diesel(n: int, sr: int, gen: np.random.Generator) -> np.ndarray:
    """Tank diesel idle: ~800 RPM fundamental (~13 Hz) + broad spectral mass."""
    t = np.arange(n) / sr
    fundamental = 13.3  # Hz
    sig = np.zeros(n)
    for h in range(1, 12):
        amp = 1.0 / h
        sig += amp * np.sin(2 * math.pi * fundamental * h * t
                            + gen.uniform(0, 2 * math.pi))
    # Broadband mechanical noise floor
    sig += gen.standard_normal(n) * 0.2
    sig /= rms(sig) + 1e-8
    return sig.astype(np.float32)


def gen_drone_hum(n: int, sr: int, gen: np.random.Generator) -> np.ndarray:
    """Quadrotor drone: motor whine ~3-4 kHz + prop blade pitch ~200 Hz."""
    t = np.arange(n) / sr
    motor = np.sin(2 * math.pi * 3400 * t + gen.uniform(0, 2 * math.pi))
    blade = np.sin(2 * math.pi * 220  * t + gen.uniform(0, 2 * math.pi))
    sig = 0.7 * motor + 0.3 * blade + gen.standard_normal(n) * 0.05
    sig /= rms(sig) + 1e-8
    return sig.astype(np.float32)


# ===========================================================================
# Noise generators – Class 1: Non-Stationary
# ===========================================================================

def gen_rotor_bpf_sweep(n: int, sr: int, gen: np.random.Generator) -> np.ndarray:
    """
    Helicopter rotor BPF sweep (MI-17 style).
    MI-17 main rotor: ~5 blades, ~225 RPM => BPF ~ 18.75 Hz.
    We modulate this slowly to simulate approach / banking.
    """
    t = np.arange(n) / sr
    bpf_base = 18.75
    # RPM varies ±10% over 3 s
    rpm_mod = 1.0 + 0.1 * np.sin(2 * math.pi * 0.33 * t)
    inst_phase = 2 * math.pi * bpf_base * np.cumsum(rpm_mod) / sr
    sig = np.sin(inst_phase)
    # Add harmonic series (BPF × 1..6)
    for h in range(2, 7):
        sig += (1.0 / h) * np.sin(h * inst_phase + gen.uniform(0, 2 * math.pi))
    sig += gen.standard_normal(n) * 0.15
    sig /= rms(sig) + 1e-8
    return sig.astype(np.float32)


def gen_siren_doppler(n: int, sr: int, gen: np.random.Generator) -> np.ndarray:
    """Emergency siren Doppler sweep: 800 Hz -> 1200 Hz -> 800 Hz wail."""
    t = np.arange(n) / sr
    # Wail: sinusoidal frequency modulation at 1 Hz rate
    f_carrier = 1000.0
    f_dev = 200.0
    f_mod = 1.0
    inst_freq = f_carrier + f_dev * np.sin(2 * math.pi * f_mod * t)
    phase = 2 * math.pi * np.cumsum(inst_freq) / sr
    sig = np.sin(phase)
    # Yelp overtone
    sig += 0.3 * np.sin(2 * phase + gen.uniform(0, 2 * math.pi))
    sig += gen.standard_normal(n) * 0.05
    sig /= rms(sig) + 1e-8
    return sig.astype(np.float32)


def gen_vehicle_rpm_ramp(n: int, sr: int, gen: np.random.Generator) -> np.ndarray:
    """Vehicle accelerating: RPM ramp 1000->4000 over clip duration."""
    t = np.arange(n) / sr
    # Engine fundamental sweeps from ~16 Hz to ~66 Hz
    f_start, f_end = 16.0, 66.0
    f_inst = f_start + (f_end - f_start) * (t / DURATION)
    phase = 2 * math.pi * np.cumsum(f_inst) / sr
    sig = np.sin(phase)
    for h in range(2, 8):
        sig += (1.0 / h) * np.sin(h * phase + gen.uniform(0, 2 * math.pi))
    sig += gen.standard_normal(n) * 0.1
    sig /= rms(sig) + 1e-8
    return sig.astype(np.float32)


# ===========================================================================
# Noise generators – Class 2: Impulsive
# ===========================================================================

def gen_rifle_shot(n: int, sr: int, gen: np.random.Generator,
                   n_shots: int = 1) -> tuple[np.ndarray, list[float]]:
    """
    Single-shot rifle (AK-47 style): exponentially decaying broadband shockwave.
    Returns (signal, impulse_timestamps_seconds).
    """
    sig = np.zeros(n, dtype=np.float32)
    timestamps = []
    for _ in range(n_shots):
        t_shot = gen.uniform(0.2, DURATION - 0.3)
        sample = int(t_shot * sr)
        duration_samps = int(0.012 * sr)   # 12 ms decay
        decay = np.exp(-np.arange(duration_samps) / (0.002 * sr))
        noise = gen.standard_normal(duration_samps).astype(np.float32)
        blast = (decay * noise).astype(np.float32)
        end = min(sample + duration_samps, n)
        sig[sample:end] += blast[:end - sample] * 3.0
        timestamps.append(round(t_shot, 4))
    sig = np.clip(sig, -1.0, 1.0)
    return sig, sorted(timestamps)


def gen_machinegun_burst(n: int, sr: int,
                          gen: np.random.Generator) -> tuple[np.ndarray, list[float]]:
    """
    Machine-gun burst: RPM ~600 rounds/min => ~10 shots/s.
    5-shot burst starting at random offset.
    """
    rate = 10.0   # shots per second
    n_shots = 5
    start_t = gen.uniform(0.1, 0.5)
    sig = np.zeros(n, dtype=np.float32)
    timestamps = []
    for i in range(n_shots):
        t_shot = start_t + i / rate
        if t_shot >= DURATION:
            break
        sample = int(t_shot * sr)
        dur = int(0.008 * sr)
        decay = np.exp(-np.arange(dur) / (0.001 * sr))
        noise = gen.standard_normal(dur).astype(np.float32)
        blast = (decay * noise * 2.5).astype(np.float32)
        end = min(sample + dur, n)
        sig[sample:end] += blast[:end - sample]
        timestamps.append(round(t_shot, 4))
    sig = np.clip(sig, -1.0, 1.0)
    return sig, sorted(timestamps)


def gen_mortar_blast(n: int, sr: int,
                      gen: np.random.Generator) -> tuple[np.ndarray, list[float]]:
    """
    Mortar / artillery: a longer, lower-frequency shockwave with trailing reverb.
    """
    sig = np.zeros(n, dtype=np.float32)
    t_blast = gen.uniform(0.3, 1.5)
    sample = int(t_blast * sr)
    dur_main = int(0.05 * sr)    # 50 ms main blast
    dur_reverb = int(0.3 * sr)   # 300 ms reverb tail

    decay_main = np.exp(-np.arange(dur_main) / (0.005 * sr))
    noise_main = gen.standard_normal(dur_main).astype(np.float32)
    blast = (decay_main * noise_main * 4.0).astype(np.float32)
    end = min(sample + dur_main, n)
    sig[sample:end] += blast[:end - sample]

    # Reverb tail
    rev_start = sample + int(0.02 * sr)
    decay_rev = np.exp(-np.arange(dur_reverb) / (0.06 * sr))
    noise_rev = gen.standard_normal(dur_reverb).astype(np.float32)
    reverb = (decay_rev * noise_rev * 0.6).astype(np.float32)
    end_rev = min(rev_start + dur_reverb, n)
    sig[rev_start:end_rev] += reverb[:end_rev - rev_start]

    sig = np.clip(sig, -1.0, 1.0)
    return sig, [round(t_blast, 4)]


# ===========================================================================
# Noise class dispatch table
# ===========================================================================

CLASS_GENERATORS = {
    # class_id: list of (name, generator_fn, impulsive)
    0: [
        ("field_gen_50hz",   lambda n, sr, g: (gen_field_generator(n, sr, g, 50.0), []),  False),
        ("field_gen_60hz",   lambda n, sr, g: (gen_field_generator(n, sr, g, 60.0), []),  False),
        ("tank_diesel",      lambda n, sr, g: (gen_tank_diesel(n, sr, g), []),             False),
        ("drone_hum",        lambda n, sr, g: (gen_drone_hum(n, sr, g), []),               False),
    ],
    1: [
        ("rotor_bpf_sweep",  lambda n, sr, g: (gen_rotor_bpf_sweep(n, sr, g), []),         False),
        ("siren_doppler",    lambda n, sr, g: (gen_siren_doppler(n, sr, g), []),            False),
        ("vehicle_rpm_ramp", lambda n, sr, g: (gen_vehicle_rpm_ramp(n, sr, g), []),         False),
    ],
    2: [
        ("rifle_shot",      gen_rifle_shot,    True),
        ("machinegun_burst", gen_machinegun_burst, True),
        ("mortar_blast",    gen_mortar_blast,   True),
    ],
}


# ===========================================================================
# Dataset generation main loop
# ===========================================================================

def generate_dataset(out_dir: str, clips_per_combo: int = 3,
                     seed: int = 42) -> None:
    random.seed(seed)
    gen = rng_seed(seed)

    noisy_dir = os.path.join(out_dir, "noisy")
    clean_dir = os.path.join(out_dir, "clean")
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    meta_rows = []
    file_idx = 0

    print(f"\n{'='*64}")
    print(f"  DeepFilterNet Defence Dataset Generator")
    print(f"  SR={SR} Hz | Duration={DURATION}s | {N_SAMPLES} samples/clip")
    print(f"  SNR grid: {SNR_GRID} dB | clips/combo: {clips_per_combo}")
    print(f"  Output: {os.path.abspath(out_dir)}")
    print(f"{'='*64}\n")

    for class_id, generators in CLASS_GENERATORS.items():
        class_name = ["stationary", "non_stationary", "impulsive"][class_id]
        print(f"[CLASS {class_id}: {class_name.upper()}]")

        for gen_info in generators:
            noise_name, noise_fn, is_impulsive = gen_info

            for snr in SNR_GRID:
                for clip_i in range(clips_per_combo):
                    local_seed = seed + file_idx * 97
                    local_gen = rng_seed(local_seed)

                    # -- Generate clean speech
                    speech = gen_clean_speech(N_SAMPLES, SR, local_gen)

                    # -- Generate noise (returns tuple: signal, timestamps)
                    noise_raw, impulse_ts = noise_fn(N_SAMPLES, SR, local_gen)

                    # -- Mix at target SNR
                    noisy = mix_at_snr(speech, noise_raw, snr)
                    noisy = np.clip(noisy, -1.0, 1.0).astype(np.float32)

                    # -- File naming
                    fname = (f"cls{class_id}_{noise_name}_snr{snr:+d}dB"
                             f"_clip{clip_i:02d}_{file_idx:04d}")
                    noisy_path = os.path.join(noisy_dir, fname + ".wav")
                    clean_path = os.path.join(clean_dir, fname + ".wav")

                    save_wav(noisy_path, noisy, SR)
                    save_wav(clean_path, speech, SR)

                    meta_rows.append({
                        "file_id":            fname,
                        "class_id":           class_id,
                        "class_name":         class_name,
                        "noise_type":         noise_name,
                        "snr_db":             snr,
                        "clip_index":         clip_i,
                        "impulse_timestamps": str(impulse_ts),
                        "n_impulses":         len(impulse_ts),
                        "noisy_path":         noisy_path,
                        "clean_path":         clean_path,
                    })

                    file_idx += 1

            print(f"  [OK] {noise_name:<22}  ({clips_per_combo * len(SNR_GRID)} files)")

    # -- Write metadata CSV
    csv_path = os.path.join(out_dir, "metadata.csv")
    fieldnames = list(meta_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(meta_rows)

    # -- Write stats JSON
    from collections import Counter
    class_counts = Counter(r["class_id"] for r in meta_rows)
    snr_counts   = Counter(r["snr_db"]   for r in meta_rows)
    stats = {
        "total_clips":     file_idx,
        "total_duration_s": round(file_idx * DURATION, 1),
        "sr_hz":           SR,
        "snr_grid":        SNR_GRID,
        "class_breakdown": {str(k): v for k, v in sorted(class_counts.items())},
        "snr_breakdown":   {str(k): v for k, v in sorted(snr_counts.items())},
        "clips_per_combo": clips_per_combo,
        "seed":            seed,
    }
    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    # -- Summary
    print(f"\n{'='*64}")
    print(f"  Dataset generation COMPLETE")
    print(f"  Total clips   : {file_idx}")
    print(f"  Total duration: {stats['total_duration_s']} s  "
          f"({stats['total_duration_s']/60:.1f} min)")
    print(f"  Metadata CSV  : {csv_path}")
    print(f"  Stats JSON    : {stats_path}")
    print(f"{'='*64}\n")


# ===========================================================================
# CLI entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic military-noise dataset for DeepFilterNet Defence."
    )
    parser.add_argument(
        "--out_dir", default="data/defense_dataset",
        help="Output directory (default: data/defense_dataset)"
    )
    parser.add_argument(
        "--clips_per_combo", type=int, default=3,
        help="Number of clips per (noise_type × snr) combination (default: 3)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()
    generate_dataset(args.out_dir, args.clips_per_combo, args.seed)


if __name__ == "__main__":
    main()

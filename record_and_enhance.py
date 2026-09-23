#!/usr/bin/env python3
"""Record Live Voice and Automatically Enhance with EchoShield
Records directly from your physical microphone, filters out noise, and saves both
the original noisy recording and the enhanced clean voice to WAV files.

Usage:
    python record_and_enhance.py
    python record_and_enhance.py --duration 5
    python record_and_enhance.py --device "Realtek"
"""

import sys
import os

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import time
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

# Add EchoShield to python path
project_root = Path(__file__).resolve().parent
df_path = project_root / "EchoShield"
if str(df_path) not in sys.path:
    sys.path.insert(0, str(df_path))

from df.enhance import init_df, enhance


def find_physical_microphone(preferred_device=None):
    """Find the real physical microphone, avoiding virtual audio cables."""
    all_devs = sd.query_devices()

    if preferred_device is not None:
        if str(preferred_device).isdigit():
            return int(preferred_device)
        for i, d in enumerate(all_devs):
            if d['max_input_channels'] > 0 and str(preferred_device).lower() in d['name'].lower():
                return i

    # Look for Realtek or physical microphones first
    for i, d in enumerate(all_devs):
        name = d['name'].lower()
        if d['max_input_channels'] > 0 and ('cable' not in name and 'virtual' not in name):
            if 'realtek' in name or 'array' in name or 'mic' in name or 'headset' in name:
                return i

    # Fallback to any non-cable input
    for i, d in enumerate(all_devs):
        if d['max_input_channels'] > 0 and 'cable' not in d['name'].lower():
            return i

    return sd.default.device[0]


def record_live(duration=None, samplerate=48000, device=None):
    """Record audio from physical microphone with live VU meter."""
    dev_id = find_physical_microphone(device)
    dev_info = sd.query_devices(dev_id)

    print("\n" + "=" * 65)
    print("🎙️ LIVE MICROPHONE RECORDING")
    print("=" * 65)
    print(f"[*] Selected Mic : [{dev_id}] {dev_info['name']}")
    print(f"[*] Sample Rate  : {samplerate} Hz")
    print("-" * 65)

    recording = []
    running = True

    def callback(indata, frames, time_info, status):
        if running:
            mono = indata[:, 0].copy()
            recording.append(mono)
            rms = np.sqrt(np.mean(mono**2) + 1e-12)
            db = 20.0 * np.log10(rms)
            filled = max(0, min(15, int((db + 50) / 45 * 15)))
            bar = "█" * filled + "░" * (15 - filled)
            sys.stdout.write(f"\r  Mic Level: [{bar}] {db:5.1f} dB  (Speaking...)   ")
            sys.stdout.flush()

    if duration:
        print(f"[*] Recording for {duration} seconds (Speak now!)...")
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', device=dev_id, callback=callback):
            time.sleep(duration)
            running = False
        print("\n\n[✓] Recording complete!")
    else:
        print("[*] Speak now! Press ENTER when finished to stop recording:")
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', device=dev_id, callback=callback):
            input("\n\n>>> Press [ENTER] to STOP recording <<<\n")
            running = False
        print("\n[✓] Recording stopped!")

    if not recording:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(recording, axis=0)
    return audio


def list_devices():
    """Print available input audio devices."""
    devices = sd.query_devices()
    print("\n" + "=" * 65)
    print("AVAILABLE MICROPHONES / INPUT DEVICES")
    print("=" * 65)
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            api = sd.query_hostapis(d['hostapi'])['name']
            print(f"[{i:2d}] {d['name']} ({api})")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Record live voice and enhance with EchoShield")
    parser.add_argument("--list-devices", "-l", action="store_true", help="List all microphone input devices.")
    parser.add_argument("--duration", "-d", type=float, default=None, help="Recording duration in seconds (optional).")
    parser.add_argument("--output-dir", "-o", default="results", help="Directory to save the recorded files.")
    parser.add_argument("--model", "-m", default="DeepFilterNet3", help="Model to use (EchoShield, EchoShield2, EchoShield3).")
    parser.add_argument("--pf", action="store_true", help="Enable post-filter.")
    parser.add_argument("--device", "-i", default=None, help="Input microphone device ID or name substring.")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Record live audio
    audio_np = record_live(duration=args.duration, samplerate=48000, device=args.device)
    if len(audio_np) == 0:
        print("[!] No audio recorded.")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"recording_raw_{timestamp}.wav"
    clean_path = out_dir / f"recording_clean_{timestamp}.wav"

    # 2. Save raw recording
    sf.write(str(raw_path), audio_np, 48000)
    print(f"\n[💾] Saved original raw recording to: {raw_path}")

    # 3. AI Enhancement
    print(f"[*] Enhancing with {args.model} AI Speech Filter...")
    model, df_state, _, _ = init_df(model_base_dir=args.model, post_filter=args.pf, log_level="error")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    audio_t = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)
    start_t = time.time()
    clean_t = enhance(model, df_state, audio_t, pad=True)
    elapsed = time.time() - start_t

    clean_np = clean_t.squeeze(0).numpy()
    sf.write(str(clean_path), clean_np, 48000)

    print(f"[✨] Processed {len(audio_np)/48000:.1f}s audio in {elapsed:.2f}s!")
    print(f"[💾] Saved enhanced clean speech to: {clean_path}\n")
    print("=" * 65)
    print(f"👉 Original (Noisy): {raw_path}")
    print(f"👉 Enhanced (Clean): {clean_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

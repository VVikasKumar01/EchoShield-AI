#!/usr/bin/env python3
"""DeepFilterNet Live Real-Time Speech Enhancement
Uses decoupled non-blocking InputStream and OutputStream
to guarantee seamless cross-device audio routing with zero dropouts.

Modes:
1. Live Calls (Zoom/Meet/Discord): Route output to "CABLE Input"
2. Self-Monitoring: Hear your clean voice in your Headphones/Speakers with --hear-myself
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
import queue
import threading
from pathlib import Path
import numpy as np
import torch
import sounddevice as sd

# Add DeepFilterNet to python path
project_root = Path(__file__).resolve().parent
df_path = project_root / "DeepFilterNet"
if str(df_path) not in sys.path:
    sys.path.insert(0, str(df_path))

from df.enhance import init_df, enhance

# Optimize PyTorch CPU inference
torch.set_num_threads(max(1, os.cpu_count() // 2))


def list_devices():
    """Print available input and output audio devices."""
    devices = sd.query_devices()
    print("\n" + "=" * 65)
    print("AVAILABLE AUDIO DEVICES")
    print("=" * 65)
    for i, d in enumerate(devices):
        api = sd.query_hostapis(d['hostapi'])['name']
        in_ch = d['max_input_channels']
        out_ch = d['max_output_channels']
        kind = "INPUT " if in_ch > 0 and out_ch == 0 else "OUTPUT" if out_ch > 0 and in_ch == 0 else "I/O   "
        print(f"[{i:2d}] {kind} | {d['name'][:35]:<35} | {api}")
    print("=" * 65 + "\n")


def parse_device_arg(arg_val, is_input=True, pair_with_id=None):
    """Resolve device ID or substring to device index."""
    if arg_val is None:
        return None
    if isinstance(arg_val, int):
        return arg_val
    if str(arg_val).isdigit():
        return int(arg_val)

    all_devs = sd.query_devices()
    target_api = None
    if pair_with_id is not None:
        target_api = all_devs[pair_with_id]['hostapi']
    else:
        def_id = sd.default.device[0] if is_input else sd.default.device[1]
        if def_id >= 0:
            target_api = all_devs[def_id]['hostapi']

    matches = []
    for idx, d in enumerate(all_devs):
        channels = d['max_input_channels'] if is_input else d['max_output_channels']
        if channels > 0 and str(arg_val).lower() in d['name'].lower():
            matches.append(idx)

    if not matches:
        return None

    # First match with the same host API as the other device
    if target_api is not None:
        for idx in matches:
            if all_devs[idx]['hostapi'] == target_api:
                return idx

    # Otherwise prefer MME (most compatible on Windows) or DirectSound
    for idx in matches:
        host_api = sd.query_hostapis(all_devs[idx]['hostapi'])['name']
        if 'MME' in host_api:
            return idx
    return matches[0]


def make_meter(db_val, min_db=-50, max_db=-5, width=10):
    if db_val <= min_db:
        filled = 0
    elif db_val >= max_db:
        filled = width
    else:
        norm = (db_val - min_db) / (max_db - min_db)
        filled = int(norm * width)
    return "█" * filled + "░" * (width - filled)


class LiveDeepFilterStream:
    def __init__(
        self,
        model_name: str = "DeepFilterNet3",
        post_filter: bool = False,
        input_device=None,
        output_device=None,
        monitor_device=None,
        atten_lim_db: float = None,
        block_size: int = 2400,  # 2400 samples = 50ms at 48kHz
        pre_buffer_blocks: int = 3,
    ):
        self.sr = 48000
        self.block_size = block_size
        self.atten_lim_db = atten_lim_db
        self.pre_buffer_blocks = pre_buffer_blocks

        print(f"[*] Initializing model: {model_name}...")
        self.model, self.df_state, self.suffix, _ = init_df(
            model_base_dir=model_name,
            post_filter=post_filter,
            log_level="error",
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.input_device = parse_device_arg(input_device, is_input=True)
        self.output_device = parse_device_arg(output_device, is_input=False, pair_with_id=self.input_device)
        self.monitor_device = parse_device_arg(monitor_device, is_input=False, pair_with_id=self.input_device) if monitor_device else None

        self.in_queue = queue.Queue(maxsize=30)
        self.out_queue = queue.Queue(maxsize=30)
        self.monitor_queue = queue.Queue(maxsize=30) if self.monitor_device is not None else None
        self.running = False

        self.last_in_db = -60.0
        self.last_out_db = -60.0
        self.last_red_db = 0.0

    def _processing_worker(self):
        while self.running:
            try:
                chunk = self.in_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                in_rms = np.sqrt(np.mean(chunk**2) + 1e-12)
                in_db = 20.0 * np.log10(in_rms)

                audio_t = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    clean_t = enhance(
                        self.model,
                        self.df_state,
                        audio_t,
                        pad=True,
                        atten_lim_db=self.atten_lim_db
                    )
                clean_chunk = clean_t.squeeze(0).numpy()

                out_rms = np.sqrt(np.mean(clean_chunk**2) + 1e-12)
                out_db = 20.0 * np.log10(out_rms)

                self.last_in_db = in_db
                self.last_out_db = out_db
                self.last_red_db = max(0.0, in_db - out_db)

                if len(clean_chunk) < self.block_size:
                    clean_chunk = np.pad(clean_chunk, (0, self.block_size - len(clean_chunk)))
                elif len(clean_chunk) > self.block_size:
                    clean_chunk = clean_chunk[:self.block_size]

                if not self.out_queue.full():
                    self.out_queue.put_nowait(clean_chunk)

                if self.monitor_queue and not self.monitor_queue.full():
                    self.monitor_queue.put_nowait(clean_chunk)

            except Exception as e:
                print(f"[Worker Error] {e}", file=sys.stderr)

    def start(self):
        self.running = True

        for _ in range(self.pre_buffer_blocks):
            self.out_queue.put(np.zeros(self.block_size, dtype=np.float32))
            if self.monitor_queue:
                self.monitor_queue.put(np.zeros(self.block_size, dtype=np.float32))

        worker = threading.Thread(target=self._processing_worker, daemon=True)
        worker.start()

        def in_callback(indata, frames, time_info, status):
            mono = indata[:, 0].copy()
            if not self.in_queue.full():
                self.in_queue.put_nowait(mono)

        def out_callback(outdata, frames, time_info, status):
            try:
                clean = self.out_queue.get_nowait()
                outdata[:, 0] = clean
                if outdata.shape[1] > 1:
                    outdata[:, 1:] = clean[:, np.newaxis]
            except queue.Empty:
                outdata.fill(0)

        def mon_callback(outdata, frames, time_info, status):
            try:
                clean = self.monitor_queue.get_nowait()
                outdata[:, 0] = clean
                if outdata.shape[1] > 1:
                    outdata[:, 1:] = clean[:, np.newaxis]
            except queue.Empty:
                outdata.fill(0)

        in_info = sd.query_devices(self.input_device) if self.input_device is not None else sd.query_devices(sd.default.device[0])
        out_info = sd.query_devices(self.output_device) if self.output_device is not None else sd.query_devices(sd.default.device[1])

        print("\n" + "=" * 68)
        print(" [LIVE] DEEPFILTERNET REAL-TIME AUDIO ENHANCEMENT ACTIVE")
        print("=" * 68)
        print(f" Input Mic   : {in_info['name']}")
        print(f" Output Target: {out_info['name']}")
        if self.monitor_device is not None:
            mon_info = sd.query_devices(self.monitor_device)
            print(f" Monitor Out : {mon_info['name']} (Headphones/Speakers)")
        print(f" Sample Rate : {self.sr} Hz | Latency: ~50 ms")
        print("-" * 68)
        print(" Live Noise Suppression Status:")
        print("=" * 68)

        # Open separate Input and Output streams to prevent PaErrorCode -9993 cross-API issues
        in_stream = sd.InputStream(
            device=self.input_device,
            samplerate=self.sr,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=in_callback
        )

        out_stream = sd.OutputStream(
            device=self.output_device,
            samplerate=self.sr,
            blocksize=self.block_size,
            channels=min(2, out_info['max_output_channels']),
            dtype="float32",
            callback=out_callback
        )

        streams = [in_stream, out_stream]

        if self.monitor_device is not None:
            mon_info = sd.query_devices(self.monitor_device)
            mon_stream = sd.OutputStream(
                device=self.monitor_device,
                samplerate=self.sr,
                blocksize=self.block_size,
                channels=min(2, mon_info['max_output_channels']),
                dtype="float32",
                callback=mon_callback
            )
            streams.append(mon_stream)

        for s in streams:
            s.start()

        try:
            while True:
                in_b = make_meter(self.last_in_db)
                out_b = make_meter(self.last_out_db)
                red = self.last_red_db
                sys.stdout.write(
                    f"\r  Mic In: [{in_b}] {self.last_in_db:5.1f} dB  "
                    f"──[Clean Out: [{out_b}] {self.last_out_db:5.1f} dB]  "
                    f"──[Suppression: -{red:4.1f} dB]   "
                )
                sys.stdout.flush()
                time.sleep(0.08)
        except KeyboardInterrupt:
            print("\n\n[!] Live enhancement stopped by user.")
        finally:
            self.running = False
            for s in streams:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="DeepFilterNet Real-Time Live Microphone Noise Suppression")
    parser.add_argument(
        "--list-devices", "-l", action="store_true", help="List all available audio input and output devices."
    )
    parser.add_argument(
        "--input-device", "-i", default=None, help="Input microphone device ID or name substring."
    )
    parser.add_argument(
        "--output-device", "-o", default=None, help="Output playback/cable device ID or name substring."
    )
    parser.add_argument(
        "--hear-myself", action="store_true", help="Play the clean filtered voice to your headphones/speakers too while streaming to cable!"
    )
    parser.add_argument(
        "--model", "-m", default="DeepFilterNet3", help="Model name (DeepFilterNet, DeepFilterNet2, DeepFilterNet3)."
    )
    parser.add_argument(
        "--pf", action="store_true", help="Enable post-filter for extra noise attenuation."
    )
    parser.add_argument(
        "--atten-lim", "-a", type=float, default=None, help="Noise attenuation limit in dB."
    )
    parser.add_argument(
        "--block-size", "-b", type=int, default=2400, help="Block size in samples (2400 = 50ms at 48kHz)."
    )

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # If --hear-myself is specified or output not given
    out_dev = args.output_device
    mon_dev = None

    if args.hear_myself and out_dev:
        mon_dev = "Speakers" # Send to speakers/headphones as monitor

    live_stream = LiveDeepFilterStream(
        model_name=args.model,
        post_filter=args.pf,
        input_device=args.input_device,
        output_device=out_dev,
        monitor_device=mon_dev,
        atten_lim_db=args.atten_lim,
        block_size=args.block_size,
    )
    live_stream.start()


if __name__ == "__main__":
    main()

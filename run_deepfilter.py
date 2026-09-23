#!/usr/bin/env python3
"""EchoShield Execution Helper Script
Allows easy execution of speech enhancement on audio files or directories.
Usage:
    python run_deepfilter.py [OPTIONS] noisy_audio.wav [noisy_audio2.wav ...]
    python run_deepfilter.py -i path/to/noisy_dir -o path/to/output_dir
"""

import sys
from pathlib import Path

# Add EchoShield to python path
project_root = Path(__file__).resolve().parent
df_path = project_root / "EchoShield"
if str(df_path) not in sys.path:
    sys.path.insert(0, str(df_path))

from df.enhance import run

if __name__ == "__main__":
    run()

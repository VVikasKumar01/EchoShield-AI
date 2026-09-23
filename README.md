<div align="center">

# 🛡️ EchoShield AI

**Real-Time Full-Band (48kHz) Speech Enhancement & Noise Suppression Engine**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-MIT)
[![Sample Rate](https://img.shields.io/badge/Audio-48kHz%20Full--Band-green.svg)](https://github.com/VVikasKumar01/EchoShield-AI)

*EchoShield is an advanced, ultra-low latency AI speech enhancement system designed to eliminate background noise, reverberation, and unwanted acoustic artifacts in real time while preserving crystal-clear voice fidelity.*

---
</div>

## 🌟 Key Features

- 🎧 **Full-Band 48kHz Audio Quality**: Studio-grade speech enhancement supporting high-fidelity 48 kHz sampling rate.
- ⚡ **Ultra-Low Latency & Real-Time Processing**: Frame-by-frame deep filtering optimized for minimal CPU/GPU overhead.
- 🌐 **Modern Interactive Web Dashboard**: Built-in dark/light UI to upload audio, record voice, inspect waveforms, and compare before/after results with one click.
- 🎙️ **Live Virtual Microphone Filter**: Stream AI-cleaned voice directly into **Zoom, Google Meet, Discord, Microsoft Teams, Skype, OBS**, and more.
- 🔴 **Record & Auto-Enhance CLI**: Instantly record live from your microphone and get side-by-side raw and enhanced WAV outputs.
- 📁 **Batch File Processing**: Clean entire directories or multiple audio files (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`) in one command.
- 🧠 **Deep Filtering Architecture**: State-of-the-art ERB filterbank + complex linear prediction + deep perceptual convolutional recurrent network.

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/VVikasKumar01/EchoShield-AI.git
cd EchoShield-AI
```

### 2. Set Up Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎯 How to Run

### Mode 1: Interactive Web Dashboard (Recommended)
Launch the modern web UI accessible from your browser:
```bash
python web_app.py
```
Or specify a custom port:
```bash
python web_app.py 7860
```
Open your browser at **`http://localhost:7860`** to:
1. Drag & drop audio files (WAV, MP3, FLAC, M4A, etc.) to enhance instantly.
2. Record directly in your browser with real-time waveform inspection.
3. Download clean, noise-free enhanced audio files.

---

### Mode 2: Record Live Voice & Auto-Enhance (CLI)
Record your voice from your physical microphone and save both raw & enhanced WAV files:
```bash
# Record for 5 seconds (default)
python record_and_enhance.py

# Record for 10 seconds
python record_and_enhance.py --duration 10

# Record using a specific microphone device
python record_and_enhance.py --duration 5 --device "Microphone Array"
```
The output files will be automatically saved in the `results/` directory:
- `results/recording_raw_<timestamp>.wav` (Original input)
- `results/recording_enhanced_<timestamp>.wav` (Cleaned voice)

---

### Mode 3: Live Real-Time Microphone Filter (Calls & Meetings)
Filter your microphone live while you speak:

#### Option A: Self-Monitoring (Listen to yourself)
Hear your filtered voice in your headphones in real-time:
```bash
python live_deepfilter.py --hear-myself
```

#### Option B: Route into Zoom / Google Meet / Discord / OBS
1. Install a virtual audio cable (e.g. [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)).
2. Run EchoShield with the virtual cable as output:
   ```bash
   python live_deepfilter.py --out-device "CABLE Input"
   ```
3. In Zoom / Discord / Google Meet / OBS, select **"CABLE Output (VB-Audio Virtual Cable)"** as your input microphone.

---

### Mode 4: Batch Process Existing Audio Files
Enhance a single file or an entire folder of recordings:
```bash
# Enhance a single file
python run_deepfilter.py input_noisy.wav

# Enhance all audio files in a directory and save to an output folder
python run_deepfilter.py -i path/to/noisy_folder -o path/to/clean_output
```

---

## 📂 Project Structure

```
EchoShield-AI/
├── EchoShield/               # Core deep filtering enhancement engine (df package)
│   ├── enhance.py            # Enhancement pipeline & model loading
│   ├── modules.py            # Neural network architectures
│   ├── io.py                 # 48kHz audio I/O & resampling utilities
│   └── config.py             # Model configuration & hyperparameters
├── models/                   # Pre-trained deep filtering models
├── web_app.py                # Modern Web UI interface (HTML5 / Audio API)
├── record_and_enhance.py     # Live microphone recording & instant enhancement tool
├── live_deepfilter.py        # Real-time streaming microphone noise filter
├── run_deepfilter.py         # Batch file enhancement CLI utility
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Hardware & Performance

| Mode | Processing Latency | Sample Rate | CPU / GPU Overhead |
|---|---|---|---|
| **Real-Time Stream** | ~10 - 20 ms | 48 kHz (Full Band) | < 15% Single CPU Core |
| **File / Web Processing**| ~10x Real-Time (Faster than playback) | 48 kHz (Full Band) | Optimized PyTorch Execution |

---

## 🤝 Contributing

Contributions are always welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License and Apache 2.0. See `LICENSE-MIT` and `LICENSE-APACHE` for more information.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/VVikasKumar01">Vikas Kumar</a>
</div>

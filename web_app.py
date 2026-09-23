#!/usr/bin/env python3
"""EchoShield Modern Web Interface
Interactive GUI for EchoShield Speech Enhancement with:
- Audio File Upload & Processing (Supports WAV, MP3, M4A, FLAC, OGG, WebM)
- In-browser Microphone Recording & Instant AI Cleaning
- Before / After Audio Players with Waveform Visualization
- Live Real-Time Microphone Filter (for Calls / Meetings)
- Batch Processing
"""

import sys
import os

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import io
import json
import time
import base64
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import numpy as np
import soundfile as sf
import torch

# Add DeepFilterNet to python path
project_root = Path(__file__).resolve().parent
df_path = project_root / "EchoShield"
if str(df_path) not in sys.path:
    sys.path.insert(0, str(df_path))

from df.enhance import init_df, enhance
from df.io import load_audio, save_audio, resample

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# Model Cache
MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name="DeepFilterNet3", post_filter=False):
    key = (model_name, post_filter)
    if key not in MODELS:
        print(f"[*] Loading model {model_name} (pf={post_filter})...")
        model, df_state, suffix, epoch = init_df(
            model_base_dir=model_name,
            post_filter=post_filter,
            log_level="error",
        )
        model = model.to(DEVICE)
        model.eval()
        MODELS[key] = (model, df_state, suffix)
    return MODELS[key]


# Live Stream Controller
class LiveManager:
    def __init__(self):
        self.stream = None
        self.is_running = False
        self.in_device = None
        self.out_device = None
        self.stats = {"frames": 0, "status": "Stopped"}

    def start(self, in_dev=None, out_dev=None, model_name="DeepFilterNet3", post_filter=False):
        if self.is_running:
            return True, "Already running"
        if not SOUNDDEVICE_AVAILABLE:
            return False, "sounddevice library is not available"

        try:
            from live_deepfilter import LiveDeepFilterStream
            def run_stream():
                try:
                    self.live_stream = LiveDeepFilterStream(
                        model_name=model_name,
                        post_filter=post_filter,
                        input_device=in_dev,
                        output_device=out_dev,
                        block_size=2400,
                    )
                    self.is_running = True
                    self.stats["status"] = "Running"
                    self.live_stream.start()
                except Exception as e:
                    print(f"Live stream error: {e}", file=sys.stderr)
                finally:
                    self.is_running = False
                    self.stats["status"] = "Stopped"

            self.thread = threading.Thread(target=run_stream, daemon=True)
            self.thread.start()
            time.sleep(0.5)
            return True, "Live stream started"
        except Exception as e:
            return False, str(e)

    def stop(self):
        self.is_running = False
        self.stats["status"] = "Stopped"
        return True, "Stream stopped"


LIVE_MGR = LiveManager()


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepFilterNet — Real-Time Speech Enhancement</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0a0d14;
            --bg-card: rgba(18, 24, 38, 0.7);
            --bg-card-hover: rgba(26, 35, 56, 0.85);
            --border: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(56, 189, 248, 0.3);
            --accent: #38bdf8;
            --accent-purple: #818cf8;
            --accent-green: #34d399;
            --accent-pink: #f472b6;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-dim: #64748b;
            --radius: 16px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(129, 140, 248, 0.08) 0%, transparent 40%);
            background-attachment: fixed;
        }

        header {
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(12px);
            background: rgba(10, 13, 20, 0.8);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-badge {
            background: linear-gradient(135deg, var(--accent), var(--accent-purple));
            color: #000;
            font-weight: 800;
            font-size: 1.1rem;
            width: 38px;
            height: 38px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
        }

        .logo-title {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #ffffff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-tags {
            display: flex;
            gap: 8px;
        }

        .tag {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-secondary);
        }

        .tag.active {
            background: rgba(52, 211, 153, 0.1);
            color: var(--accent-green);
            border-color: rgba(52, 211, 153, 0.3);
        }

        main {
            max-width: 1200px;
            width: 100%;
            margin: 2rem auto;
            padding: 0 2rem;
            flex: 1;
        }

        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 2rem;
            background: rgba(255, 255, 255, 0.03);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid var(--border);
            width: fit-content;
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 0.9rem;
            font-weight: 600;
            padding: 8px 18px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .tab-btn:hover {
            color: var(--text-primary);
        }

        .tab-btn.active {
            background: var(--accent);
            color: #000;
            box-shadow: 0 2px 10px rgba(56, 189, 248, 0.3);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 860px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.75rem;
            backdrop-filter: blur(16px);
            transition: all 0.3s ease;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        }

        .card:hover {
            border-color: rgba(255, 255, 255, 0.14);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .drop-zone {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 2.2rem 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: rgba(255, 255, 255, 0.02);
            margin-bottom: 1.25rem;
        }

        .drop-zone:hover, .drop-zone.dragover {
            border-color: var(--accent);
            background: rgba(56, 189, 248, 0.05);
        }

        .drop-icon {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
        }

        .drop-text {
            font-size: 0.95rem;
            color: var(--text-secondary);
        }

        .drop-hint {
            font-size: 0.8rem;
            color: var(--text-dim);
            margin-top: 4px;
        }

        .btn-row {
            display: flex;
            gap: 10px;
            margin-bottom: 1.25rem;
        }

        .btn {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border);
            color: var(--text-primary);
            font-size: 0.9rem;
            font-weight: 600;
            padding: 10px 16px;
            border-radius: 10px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.2s ease;
            flex: 1;
        }

        .btn:hover {
            background: rgba(255, 255, 255, 0.12);
            transform: translateY(-1px);
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent), #0284c7);
            color: #000;
            border: none;
            box-shadow: 0 4px 15px rgba(56, 189, 248, 0.35);
        }

        .btn-primary:hover {
            background: linear-gradient(135deg, #7dd3fc, var(--accent));
            box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5);
        }

        .btn-rec.recording {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            border-color: rgba(239, 68, 68, 0.4);
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.03); }
        }

        .form-group {
            margin-bottom: 1.25rem;
        }

        .form-label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }

        .form-select, .form-input {
            width: 100%;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 10px 14px;
            border-radius: 10px;
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.2s ease;
        }

        .form-select:focus, .form-input:focus {
            border-color: var(--accent);
        }

        .player-box {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        .player-label {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .player-label.noisy {
            color: #f87171;
        }

        .player-label.clean {
            color: var(--accent-green);
        }

        audio {
            width: 100%;
            height: 40px;
            margin-top: 6px;
            border-radius: 8px;
        }

        .live-status-card {
            text-align: center;
            padding: 2.5rem 1.5rem;
        }

        .pulse-indicator {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            background: #64748b;
        }

        .pulse-indicator.active {
            background: var(--accent-green);
            box-shadow: 0 0 12px var(--accent-green);
            animation: pulse-ring 1.5s infinite;
        }

        @keyframes pulse-ring {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.2); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.8; }
        }

        .instructions {
            background: rgba(56, 189, 248, 0.05);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            font-size: 0.85rem;
            line-height: 1.5;
            color: var(--text-secondary);
            margin-top: 1.25rem;
        }

        .instructions strong {
            color: var(--accent);
        }

        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(0,0,0,0.3);
            border-radius: 50%;
            border-top-color: #000;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        footer {
            text-align: center;
            padding: 1.5rem;
            font-size: 0.8rem;
            color: var(--text-dim);
            border-top: 1px solid var(--border);
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <div class="logo-group">
                <div class="logo-badge">DF</div>
                <div class="logo-title">DeepFilterNet 3.0</div>
            </div>
            <div class="header-tags">
                <span class="tag active">● AI Model Ready</span>
                <span class="tag">48 kHz Full-Band</span>
                <span class="tag">Low Complexity</span>
            </div>
        </div>
    </header>

    <main>
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('file-tab')">🎙️ Enhance Audio / Record</button>
            <button class="tab-btn" onclick="switchTab('live-tab')">📞 Live Call Filter (Mic)</button>
            <button class="tab-btn" onclick="switchTab('batch-tab')">📁 Batch Folder</button>
        </div>

        <!-- TAB 1: FILE & MIC ENHANCEMENT -->
        <div id="file-tab" class="tab-content active">
            <div class="grid-2">
                <!-- Input Controls -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">1. Input Audio</div>
                    </div>

                    <div class="drop-zone" id="dropZone" onclick="document.getElementById('audioFileInput').click()">
                        <div class="drop-icon">🎵</div>
                        <div class="drop-text">Click or drag & drop noisy audio (.wav, .mp3, .m4a, .flac)</div>
                        <div class="drop-hint" id="fileChosenHint">No file selected</div>
                    </div>
                    <input type="file" id="audioFileInput" accept="audio/*" style="display:none" onchange="onFileSelected(this)">

                    <div class="btn-row">
                        <button class="btn btn-rec" id="recBtn" onclick="toggleRecord()">
                            <span id="recDot">●</span> <span id="recText">Record from Mic</span>
                        </button>
                        <button class="btn" onclick="loadSampleAudio()">
                            ⚡ Load Sample Audio
                        </button>
                    </div>

                    <div class="form-group">
                        <label class="form-label">Enhancement Model</label>
                        <select class="form-select" id="modelSelect">
                            <option value="DeepFilterNet3" selected>DeepFilterNet3 (Multi-Frame, Best Quality)</option>
                            <option value="DeepFilterNet2">DeepFilterNet2 (Fast & Lightweight)</option>
                            <option value="DeepFilterNet">DeepFilterNet (Original)</option>
                        </select>
                    </div>

                    <div class="form-group" style="display:flex; justify-content:space-between; align-items:center;">
                        <label class="form-label" style="margin:0;">Enable Post-Filter (Over-attenuate heavy background noise)</label>
                        <input type="checkbox" id="pfCheckbox" style="transform:scale(1.3); cursor:pointer;">
                    </div>

                    <button class="btn btn-primary" id="enhanceBtn" style="width:100%; margin-top:1rem; padding:14px;" onclick="processAudio()">
                        ✨ Enhance Audio with DeepFilterNet
                    </button>
                </div>

                <!-- Output Player & Comparison -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">2. Before & After Comparison</div>
                    </div>

                    <div class="player-box">
                        <div class="player-label noisy">
                            <span>🔴 Original Audio (Noisy)</span>
                        </div>
                        <audio id="noisyPlayer" controls></audio>
                    </div>

                    <div class="player-box">
                        <div class="player-label clean">
                            <span>🟢 AI Enhanced Audio (Clean Speech)</span>
                            <span id="procTimeBadge" style="font-size:0.75rem; color:var(--text-dim);"></span>
                        </div>
                        <audio id="cleanPlayer" controls></audio>
                    </div>

                    <a id="downloadBtn" style="display:none; text-decoration:none;" download="enhanced_clean_audio.wav">
                        <button class="btn btn-primary" style="width:100%; margin-top:0.5rem; padding:12px;">
                            ⬇️ Download Clean Audio (.wav)
                        </button>
                    </a>

                    <div class="instructions" style="margin-top:1.5rem;">
                        <strong>💡 How it works:</strong> DeepFilterNet computes real-time ERB filterbank energy alongside complex spectrogram filtering, suppressing non-speech acoustic artifacts while preserving full voice timber.
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: LIVE CALL FILTER -->
        <div id="live-tab" class="tab-content">
            <div class="card" style="max-width: 800px; margin: 0 auto;">
                <div class="card-header">
                    <div class="card-title">Live Real-Time Microphone Filter (For Calls & Meetings)</div>
                </div>

                <div class="grid-2">
                    <div class="form-group">
                        <label class="form-label">Microphone (Input Device)</label>
                        <select class="form-select" id="liveInDevice"></select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Output Device / Virtual Cable</label>
                        <select class="form-select" id="liveOutDevice"></select>
                    </div>
                </div>

                <div class="live-status-card">
                    <div style="font-size:1.2rem; font-weight:700; margin-bottom:1rem;">
                        <span class="pulse-indicator" id="livePulse"></span>
                        <span id="liveStatusText">Engine Idle</span>
                    </div>
                    <button class="btn btn-primary" id="liveToggleBtn" style="padding:14px 32px; font-size:1rem;" onclick="toggleLiveStream()">
                        🚀 Start Live Real-Time Suppression
                    </button>
                </div>

                <div class="instructions">
                    <strong>📞 Usage with Zoom / Google Meet / Discord / Teams:</strong><br>
                    1. Install the free <a href="https://vb-audio.com/Cable/" target="_blank" style="color:var(--accent);">VB-Audio Virtual Cable</a> on Windows.<br>
                    2. Set <em>Output Device</em> above to <strong>"CABLE Input"</strong> and click Start.<br>
                    3. In your call app, choose <strong>"CABLE Output"</strong> as your Microphone!
                </div>
            </div>
        </div>

        <!-- TAB 3: BATCH PROCESSING -->
        <div id="batch-tab" class="tab-content">
            <div class="card" style="max-width: 800px; margin: 0 auto;">
                <div class="card-header">
                    <div class="card-title">📁 Batch Directory Processing</div>
                </div>
                <div class="form-group">
                    <label class="form-label">Input Directory containing .wav files</label>
                    <input type="text" class="form-input" id="batchInDir" value="assets">
                </div>
                <div class="form-group">
                    <label class="form-label">Output Directory</label>
                    <input type="text" class="form-input" id="batchOutDir" value="results/batch_output">
                </div>
                <button class="btn btn-primary" style="width:100%; padding:14px;" onclick="runBatchProcess()">
                    ⚡ Process All Audio Files
                </button>
                <div id="batchResultText" style="margin-top:1rem; font-family:'JetBrains Mono'; font-size:0.85rem; color:var(--text-secondary); white-space:pre-wrap;"></div>
            </div>
        </div>
    </main>

    <footer>
        DeepFilterNet Speech Enhancement • Low Complexity Real-Time Full-Band DSP Framework
    </footer>

    <script>
        let currentAudioBlob = null;
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
            if (tabId === 'live-tab') loadLiveDevices();
        }

        // Convert any audio blob to standard 48kHz WAV PCM Float32/Int16
        async function convertToWav48k(blob) {
            const arrayBuffer = await blob.arrayBuffer();
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
            
            const numChannels = audioBuffer.numberOfChannels;
            const length = audioBuffer.length;
            const sampleRate = 48000;
            
            // Mix to mono
            const channelData = new Float32Array(length);
            for (let c = 0; c < numChannels; c++) {
                const cData = audioBuffer.getChannelData(c);
                for (let i = 0; i < length; i++) {
                    channelData[i] += cData[i] / numChannels;
                }
            }
            
            // Encode WAV (16-bit PCM)
            const buffer = new ArrayBuffer(44 + length * 2);
            const view = new DataView(buffer);
            
            function writeString(offset, string) {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            }
            
            writeString(0, 'RIFF');
            view.setUint32(4, 36 + length * 2, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true); // PCM
            view.setUint16(22, 1, true); // Mono
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            writeString(36, 'data');
            view.setUint32(40, length * 2, true);
            
            let offset = 44;
            for (let i = 0; i < length; i++) {
                let s = Math.max(-1, Math.min(1, channelData[i]));
                view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                offset += 2;
            }
            
            return new Blob([buffer], { type: 'audio/wav' });
        }

        // Drag & Drop
        const dropZone = document.getElementById('dropZone');
        ['dragenter', 'dragover'].forEach(n => dropZone.addEventListener(n, e => { e.preventDefault(); dropZone.classList.add('dragover'); }));
        ['dragleave', 'drop'].forEach(n => dropZone.addEventListener(n, e => { e.preventDefault(); dropZone.classList.remove('dragover'); }));
        dropZone.addEventListener('drop', e => {
            if (e.dataTransfer.files.length) {
                handleAudioFile(e.dataTransfer.files[0]);
            }
        });

        function onFileSelected(input) {
            if (input.files.length) {
                handleAudioFile(input.files[0]);
            }
        }

        async function handleAudioFile(file) {
            try {
                document.getElementById('fileChosenHint').innerText = `Converting: ${file.name}...`;
                currentAudioBlob = await convertToWav48k(file);
                document.getElementById('fileChosenHint').innerText = `Ready: ${file.name}`;
                const url = URL.createObjectURL(currentAudioBlob);
                document.getElementById('noisyPlayer').src = url;
            } catch(e) {
                alert('Could not decode audio file: ' + e);
                document.getElementById('fileChosenHint').innerText = 'Decoding failed';
            }
        }

        // Mic Recording
        async function toggleRecord() {
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.onstop = async () => {
                        document.getElementById('fileChosenHint').innerText = 'Processing recording...';
                        const rawBlob = new Blob(audioChunks);
                        currentAudioBlob = await convertToWav48k(rawBlob);
                        const url = URL.createObjectURL(currentAudioBlob);
                        document.getElementById('noisyPlayer').src = url;
                        document.getElementById('fileChosenHint').innerText = 'Microphone Recording ready (48 kHz WAV)';
                    };
                    mediaRecorder.start();
                    isRecording = true;
                    document.getElementById('recBtn').classList.add('recording');
                    document.getElementById('recText').innerText = 'Stop Recording';
                } catch(err) {
                    alert('Microphone access denied: ' + err);
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                document.getElementById('recBtn').classList.remove('recording');
                document.getElementById('recText').innerText = 'Record from Mic';
            }
        }

        // Load Sample
        async function loadSampleAudio() {
            const res = await fetch('/api/sample');
            const blob = await res.blob();
            currentAudioBlob = blob;
            const url = URL.createObjectURL(blob);
            document.getElementById('noisyPlayer').src = url;
            document.getElementById('fileChosenHint').innerText = 'Loaded sample: noisy_snr0.wav';
        }

        // Process Audio
        async function processAudio() {
            if (!currentAudioBlob) {
                alert('Please upload an audio file or record from microphone first!');
                return;
            }

            const btn = document.getElementById('enhanceBtn');
            btn.innerHTML = '<span class="spinner"></span> Enhancing with DeepFilterNet...';
            btn.disabled = true;

            try {
                // Ensure WAV 48k format
                let wavBlob = currentAudioBlob;
                if (wavBlob.type !== 'audio/wav') {
                    wavBlob = await convertToWav48k(currentAudioBlob);
                }

                const startTime = performance.now();
                const res = await fetch('/api/enhance', {
                    method: 'POST',
                    headers: {
                        'X-Model': document.getElementById('modelSelect').value,
                        'X-PF': document.getElementById('pfCheckbox').checked ? '1' : '0',
                        'Content-Type': 'audio/wav'
                    },
                    body: wavBlob
                });

                if (!res.ok) {
                    const err = await res.text();
                    alert('Processing error: ' + err);
                    return;
                }

                const blob = await res.blob();
                const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
                const url = URL.createObjectURL(blob);
                document.getElementById('cleanPlayer').src = url;
                document.getElementById('procTimeBadge').innerText = `Processed in ${elapsed}s`;

                const dl = document.getElementById('downloadBtn');
                dl.href = url;
                dl.style.display = 'block';

                document.getElementById('cleanPlayer').play();
            } catch(e) {
                alert('Enhancement failed: ' + e);
            } finally {
                btn.innerHTML = '✨ Enhance Audio with DeepFilterNet';
                btn.disabled = false;
            }
        }

        // Live Stream Controls
        async function loadLiveDevices() {
            const res = await fetch('/api/devices');
            const data = await res.json();
            const inSelect = document.getElementById('liveInDevice');
            const outSelect = document.getElementById('liveOutDevice');
            inSelect.innerHTML = '';
            outSelect.innerHTML = '';

            data.devices.forEach(d => {
                if (d.max_input_channels > 0) {
                    inSelect.innerHTML += `<option value="${d.id}">${d.id}: ${d.name}</option>`;
                }
                if (d.max_output_channels > 0) {
                    outSelect.innerHTML += `<option value="${d.id}">${d.id}: ${d.name}</option>`;
                }
            });
        }

        async function toggleLiveStream() {
            const btn = document.getElementById('liveToggleBtn');
            const pulse = document.getElementById('livePulse');
            const statusText = document.getElementById('liveStatusText');

            if (btn.innerText.includes('Start')) {
                const res = await fetch('/api/live/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        in_device: document.getElementById('liveInDevice').value,
                        out_device: document.getElementById('liveOutDevice').value,
                        model: 'DeepFilterNet3'
                    })
                });
                const data = await res.json();
                if (data.ok) {
                    btn.innerText = '🛑 Stop Live Suppression';
                    btn.style.background = '#ef4444';
                    pulse.classList.add('active');
                    statusText.innerText = '● LIVE AI Filtering Active';
                    statusText.style.color = 'var(--accent-green)';
                } else {
                    alert('Error: ' + data.message);
                }
            } else {
                await fetch('/api/live/stop', { method: 'POST' });
                btn.innerText = '🚀 Start Live Real-Time Suppression';
                btn.style.background = '';
                pulse.classList.remove('active');
                statusText.innerText = 'Engine Idle';
                statusText.style.color = '';
            }
        }

        // Batch Directory
        async function runBatchProcess() {
            const inDir = document.getElementById('batchInDir').value;
            const outDir = document.getElementById('batchOutDir').value;
            const resText = document.getElementById('batchResultText');
            resText.innerText = 'Processing batch folder...';

            const res = await fetch('/api/batch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ in_dir: inDir, out_dir: outDir })
            });
            const data = await res.json();
            resText.innerText = JSON.stringify(data, null, 2);
        }
    </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/api/sample":
            sample_path = project_root / "assets" / "noisy_snr0.wav"
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404, "Sample file not found")
        elif parsed.path == "/api/devices":
            devs = []
            if SOUNDDEVICE_AVAILABLE:
                all_devs = sd.query_devices()
                for i, d in enumerate(all_devs):
                    devs.append({
                        "id": i,
                        "name": d["name"],
                        "max_input_channels": d["max_input_channels"],
                        "max_output_channels": d["max_output_channels"],
                    })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"devices": devs}).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))

        if parsed.path == "/api/enhance":
            model_name = self.headers.get("X-Model", "DeepFilterNet3")
            pf_val = self.headers.get("X-PF", "0")
            post_filter = pf_val in ("1", "true", "True")

            audio_bytes = self.rfile.read(content_length)

            if not audio_bytes:
                self.send_error(400, "No audio payload provided")
                return

            try:
                bio = io.BytesIO(audio_bytes)
                audio_np, sr = sf.read(bio)
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)  # Mono

                audio_tensor = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)
                if sr != 48000:
                    audio_tensor = resample(audio_tensor, sr, 48000)

                model, df_state, _ = get_model(model_name, post_filter)
                enhanced_tensor = enhance(model, df_state, audio_tensor)

                out_bio = io.BytesIO()
                sf.write(out_bio, enhanced_tensor.squeeze(0).numpy(), 48000, format="WAV")
                out_bytes = out_bio.getvalue()

                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(out_bytes)))
                self.end_headers()
                self.wfile.write(out_bytes)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_error(500, f"Enhancement failed: {str(e)}")

        elif parsed.path == "/api/live/start":
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8")) if body else {}
            in_dev = int(data.get("in_device", 0)) if data.get("in_device") else None
            out_dev = int(data.get("out_device", 0)) if data.get("out_device") else None
            ok, msg = LIVE_MGR.start(in_dev=in_dev, out_dev=out_dev, model_name=data.get("model", "DeepFilterNet3"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": ok, "message": msg}).encode("utf-8"))

        elif parsed.path == "/api/live/stop":
            ok, msg = LIVE_MGR.stop()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": ok, "message": msg}).encode("utf-8"))

        elif parsed.path == "/api/batch":
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8")) if body else {}
            in_dir = Path(data.get("in_dir", "assets"))
            out_dir = Path(data.get("out_dir", "results/batch_output"))
            out_dir.mkdir(parents=True, exist_ok=True)

            wav_files = list(in_dir.glob("*.wav"))
            processed = []
            model, df_state, _ = get_model("DeepFilterNet3", False)

            for w in wav_files:
                audio, sr = load_audio(str(w), sr=48000)
                enhanced = enhance(model, df_state, audio)
                out_file = out_dir / f"{w.stem}_enhanced.wav"
                save_audio(str(out_file), enhanced, sr=48000)
                processed.append(str(out_file))

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "Success", "files_enhanced": processed}).encode("utf-8"))

        else:
            self.send_error(404, "Not Found")


def run_server(port=7860):
    server = HTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"\n========================================================")
    print(f"[*] DEEPFILTERNET WEB APP RUNNING AT:")
    print(f"[*] http://localhost:{port}")
    print(f"[*] http://127.0.0.1:{port}")
    print(f"========================================================\n")
    server.serve_forever()


if __name__ == "__main__":
    port = 7860
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)

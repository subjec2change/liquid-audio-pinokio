# Liquid Audio

🎙️ **Liquid Audio** — A Gradio web interface for offline speech-to-text
dictation powered by **[LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B)**
(LiquidAI). Runs entirely on-device — no cloud API required.

## Overview

The app provides a single offline ASR dictation interface:

| Feature | Description |
|---------|-------------|
| 🎙️ ASR Dictation | Offline speech-to-text transcription via LFM2.5-Audio-1.5B |
| 🚔 Police Report Cleanup | Optional post-processing that formats the raw transcript as a professional police report |

---

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.x drivers (recommended), Apple Silicon (MPS), or CPU
- `cmake` ≥ 3.21 for source builds (optional)

---

### Step 1 — Download LFM2.5-Audio-1.5B

The model is loaded from `./models/LFM2.5-Audio-1.5B` by default.
See [LOCAL_MODEL_SETUP.md](LOCAL_MODEL_SETUP.md) for full instructions.

```bash
pip install huggingface-hub
huggingface-cli download LiquidAI/LFM2.5-Audio-1.5B \
    --local-dir ./models/LFM2.5-Audio-1.5B
```

Once the directory exists the app sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` automatically so no network traffic is attempted
at runtime.

---

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3 — Start the app

```bash
python app.py --no-share
```

Open your browser at `http://localhost:7860`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LFM_MODEL_PATH` | `models/LFM2.5-Audio-1.5B` | Path to the local LFM2.5-Audio-1.5B model directory (relative to `app.py` or absolute). When the directory exists it is used instead of downloading from Hugging Face, and offline mode is enforced automatically. |
| `WHISPER_MODEL_SIZE` | `base` | faster-whisper model size used as fallback when LFM is unavailable: `tiny`, `base`, `small`, `medium`, or `large-v3` |
| `WHISPER_MODEL_PATH` | `whisper-model` | Path to a local CTranslate2 Whisper model directory used as ASR fallback (relative to `app.py` or absolute). When the directory exists it is used instead of downloading from Hugging Face. |

### Using a custom LFM model path

```bash
LFM_MODEL_PATH=/opt/models/LFM2.5-Audio-1.5B python app.py --no-share
```

### Using a local faster-whisper fallback

```bash
# Convert a Whisper model to CTranslate2 format (once)
pip install faster-whisper ctranslate2
ct2-transformers-converter \
    --model openai/whisper-large-v3 \
    --output_dir whisper-model \
    --quantization int8

# The app will automatically use ./whisper-model/ as ASR fallback
python app.py --no-share
```

---

## Usage

### Offline ASR Dictation

1. Upload an audio file or record speech via the microphone
2. Optionally check **Apply Police Report Cleanup** to format the raw
   transcript as a professional police report
3. Click **Transcribe**
4. View the **Raw Transcript** and, if cleanup was requested, the
   **Cleaned Police Report**

> **Police Report Cleanup** fixes capitalisation, punctuation, and grammar;
> structures paragraphs; and applies standard law-enforcement conventions
> (24-hour times, MM/DD/YYYY dates, consistent formatting of addresses and
> case numbers) — without altering any factual content.

---

## Architecture

```
Browser → Gradio (port 7860) → app.py
                                   │
                          LFM2.5-Audio-1.5B  (local, CUDA/MPS/CPU)
                          ├── ASR: audio → raw transcript
                          └── Cleanup: raw transcript → police report
                          faster-whisper (local, fallback ASR only)
```

`app.py` loads LFM2.5-Audio-1.5B via the `liquid-audio` Python package
directly — no external server process required.

---

## Troubleshooting

### `liquid-audio` not found

```bash
pip install liquid-audio
```

### Model directory not found

Download the model first (see Step 1 above). The app will attempt to
download from Hugging Face if `LFM_MODEL_PATH` does not exist on disk.

### CUDA out-of-memory

LFM2.5-Audio-1.5B requires approximately 4 GB VRAM. Use a machine with
sufficient GPU memory or run on CPU (slower).

### Slow CPU inference

CPU inference is possible but significantly slower than GPU. Consider
using a smaller faster-whisper model (`WHISPER_MODEL_SIZE=tiny`) as
the fallback while waiting for GPU resources.

---

## Requirements

- Python 3.10+
- `gradio>=5.50.0`
- `liquid-audio`
- `torchaudio>=2.0.0`
- `faster-whisper>=1.0.0`

## License

See the original model license for LFM2.5-Audio-1.5B at
<https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B>.

# Local Model Setup Guide

This guide explains how to set up and use local models with the Liquid Audio application.

## Quick Start

The app now supports loading models from a local directory instead of downloading from Hugging Face each time.

### Default Behavior

- **Default local path**: `./models/LFM2.5-Audio-1.5B`
- **Fallback**: If local models are not found, the app automatically downloads from Hugging Face

## How to Download Models Locally

### Option 1: Using Hugging Face CLI (Recommended)

```bash
# Install the Hugging Face Hub CLI
pip install huggingface-hub

# Download the model to the default location
huggingface-cli download LiquidAI/LFM2.5-Audio-1.5B --local-dir ./models/LFM2.5-Audio-1.5B
```

### Option 2: Using Git LFS

```bash
# Make sure git-lfs is installed
git lfs install

# Clone the model repository
git clone https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B ./models/LFM2.5-Audio-1.5B
```

### Option 3: Manual Download

1. Visit https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B
2. Download all files from the repository
3. Place them in `./models/LFM2.5-Audio-1.5B` directory

## Configuration Options

You can specify the model path in three ways, with the following priority:

### 1. Command-Line Argument (Highest Priority)

```bash
python app.py --model-path /path/to/your/model
```

### 2. Environment Variable

```bash
# Linux/Mac
export MODEL_PATH=/path/to/your/model
python app.py

# Windows
set MODEL_PATH=C:\path\to\your\model
python app.py

# Or inline
MODEL_PATH=/path/to/your/model python app.py
```

### 3. Default Path (Lowest Priority)

Just run the app normally:
```bash
python app.py
```

It will look for models in `./models/LFM2.5-Audio-1.5B`

## Directory Structure

After downloading, your directory should look like this:

```
liquid-audio-pinokio/
├── app.py
├── models/
│   └── LFM2.5-Audio-1.5B/
│       ├── config.json
│       ├── model.safetensors (or pytorch_model.bin)
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       └── ... (other model files)
└── ... (other project files)
```

## Benefits of Local Models

1. **Faster startup**: No need to download models on each run
2. **Offline use**: Run the app without internet connection
3. **Version control**: Pin specific model versions
4. **Bandwidth saving**: Download once, use many times
5. **Privacy**: Keep models completely local

## Troubleshooting

### Problem: "Local model path not found"

**Solution**: Make sure the path exists and contains all necessary model files. The app will fall back to downloading from Hugging Face.

### Problem: Model loading fails

**Solution**: 
1. Verify all model files are downloaded correctly
2. Check file permissions
3. Ensure sufficient disk space
4. Try re-downloading the model

### Problem: Want to use a different model version

**Solution**: Download the specific version and specify its path:
```bash
python app.py --model-path /path/to/different/model/version
```

## Additional Notes

- The `models/` directory is excluded from git (in `.gitignore`)
- Model files can be large (1.5GB+), so ensure you have sufficient disk space
- First-time local loading may still take a few seconds to load weights into memory
- GPU is recommended for best performance

---

## ASR Model Setup (faster-whisper)

The app uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2-based) for
Automatic Speech Recognition. This runs **fully offline** — no internet access is needed at
runtime.

### Download the ASR Model (on a machine with internet access)

**Option 1: Hugging Face CLI (recommended)**

```bash
pip install huggingface-hub
huggingface-cli download Systran/faster-whisper-large-v3 \
    --local-dir ./models/faster-whisper-large-v3
```

**Option 2: Git clone**

```bash
git lfs install
git clone https://huggingface.co/Systran/faster-whisper-large-v3 \
    ./models/faster-whisper-large-v3
```

Copy the downloaded directory to the machine running the app before launching.

### Expected Directory Structure

```
liquid-audio-pinokio/
├── app.py
├── models/
│   └── faster-whisper-large-v3/
│       ├── config.json
│       ├── model.bin
│       ├── tokenizer.json
│       ├── vocabulary.txt
│       ├── preprocessor_config.json
│       ├── special_tokens_map.json
│       └── ... (other model files)
└── ...
```

### ASR Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_MODEL_PATH` | `./models/faster-whisper-large-v3` | Path to the local faster-whisper model directory |
| `ASR_DEVICE` | `auto` | Inference device: `auto`, `cpu`, or `cuda` |
| `ASR_COMPUTE_TYPE` | `float16` | Compute type: `float16`, `int8`, `int8_float16`, etc. |
| `ASR_BEAM_SIZE` | `5` | Beam size for transcription decoding |

You can also pass `--asr-model-path` as a CLI argument (highest priority):

```bash
python app.py --asr-model-path /path/to/faster-whisper-large-v3
```

### Offline Enforcement

`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set automatically as the very first action in
`app.py`, before any library imports. This prevents accidental network calls even if a dependency
attempts to fetch model metadata from Hugging Face.

If the `ASR_MODEL_PATH` directory does not exist at startup, the app exits immediately with a clear
error message explaining how to obtain the model.


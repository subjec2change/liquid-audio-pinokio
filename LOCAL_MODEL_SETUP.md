# Local Model Setup Guide

This guide explains how to download and set up **LFM2.5-Audio-1.5B** for fully
offline use with the Liquid Audio application.

## Quick Start

The app loads LFM2.5-Audio-1.5B from a local directory and automatically
enables offline mode — no Hugging Face connection is needed at runtime.

### Default Behavior

- **Default local path**: `./models/LFM2.5-Audio-1.5B` (relative to `app.py`)
- **Offline enforcement**: When the directory exists, the app sets
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` automatically.
- **Fallback**: If the local directory is not found the app falls back to
  downloading from Hugging Face.

---

## How to Download the Model

### Option 1: Using Hugging Face CLI (Recommended)

```bash
# Install the Hugging Face Hub CLI
pip install huggingface-hub

# Download the model to the default location
huggingface-cli download LiquidAI/LFM2.5-Audio-1.5B \
    --local-dir ./models/LFM2.5-Audio-1.5B
```

### Option 2: Using Git LFS

```bash
# Make sure git-lfs is installed
git lfs install

# Clone the model repository
git clone https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B \
    ./models/LFM2.5-Audio-1.5B
```

### Option 3: Manual Download

1. Visit <https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B>
2. Download all files from the repository
3. Place them in `./models/LFM2.5-Audio-1.5B`

---

## Configuration Options

The model path is resolved in the following priority order:

### 1. Environment Variable (Highest Priority)

```bash
# Linux/Mac
export LFM_MODEL_PATH=/path/to/your/model
python app.py

# Windows
set LFM_MODEL_PATH=C:\path\to\your\model
python app.py

# Inline
LFM_MODEL_PATH=/path/to/your/model python app.py
```

### 2. Default Path (Lowest Priority)

Just run the app normally:

```bash
python app.py
```

It will look for the model in `./models/LFM2.5-Audio-1.5B`.

---

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

---

## Benefits of Local Models

1. **Offline use**: Run the app without any internet connection
2. **Faster startup**: No download delay on each run
3. **Privacy**: Audio never leaves the machine
4. **Version control**: Pin a specific model version
5. **Bandwidth saving**: Download once, use many times

---

## Troubleshooting

### Problem: "Local model path not found"

**Solution**: Make sure the path exists and contains all necessary model
files.  The app will fall back to downloading from Hugging Face if the
directory is missing.

### Problem: Model loading fails

**Solution**:
1. Verify all model files are downloaded correctly
2. Check file permissions
3. Ensure sufficient disk space (model is approximately 2.5 GB)
4. Try re-downloading the model

### Problem: Want to use a different model path

**Solution**: Set the `LFM_MODEL_PATH` environment variable:

```bash
LFM_MODEL_PATH=/path/to/different/version python app.py
```

---

## Additional Notes

- The `models/` directory is excluded from git (see `.gitignore`)
- Model files are approximately 2.5 GB — ensure sufficient disk space
- GPU (CUDA or Apple Silicon MPS) is recommended for best performance;
  CPU inference works but is significantly slower

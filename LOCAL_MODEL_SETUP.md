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

The command-line argument sets the MODEL_PATH environment variable before models are loaded, which the `get_model_path()` function then reads. This ensures the documented precedence order works correctly.

### 1. Command-Line Argument (Highest Priority)

```bash
python app.py --model-path /path/to/your/model
```

This sets the `MODEL_PATH` environment variable internally before model loading begins.

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

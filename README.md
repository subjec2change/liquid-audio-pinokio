# Liquid Audio 

🎙️ **Liquid Audio - LFM2.5-Audio-1.5B** - A Gradio web interface for Liquid AI's multimodal audio model, packaged for Pinokio.

## Overview

Liquid Audio is an advanced multimodal audio model that seamlessly handles multiple tasks:
- **Speech-to-Speech Chat**: Engage in multi-turn conversations with both text and audio input/output
- **Automatic Speech Recognition (ASR)**: Convert speech to text with high accuracy
- **Text-to-Speech (TTS)**: Generate natural-sounding speech with multiple voice options

The LFM2.5-Audio-1.5B model supports interleaved text and audio generation, enabling rich, natural conversations.

## Features

- 🎙️ **Speech-to-Speech Chat**: Multi-turn conversations with audio and text
- 📝 **Automatic Speech Recognition**: Accurate speech-to-text transcription
- 🔊 **Text-to-Speech**: Multiple voice options (US/UK, Male/Female)
- 🔄 **Interleaved Output**: Generate combined text and audio responses
- 💬 **Customizable System Prompts**: Control response behavior
- 🎚️ **Voice Selection**: Choose from multiple voice profiles
- 🖥️ **Web Interface**: User-friendly Gradio interface accessible via browser
- 💾 **Chat History**: Preserve conversation context across turns

## Installation


### Manual Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download or prepare local models** (optional):
   - By default, models will be loaded from `./models/LFM2.5-Audio-1.5B`
   - You can download the model from [Hugging Face](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) and place it in this directory
   - If the local model path doesn't exist, the app will automatically download from Hugging Face

4. **Run the interface**:
```bash
# Use default local model path (./models/LFM2.5-Audio-1.5B)
python app.py

# Or specify a custom model path
python app.py --model-path /path/to/your/model

# Or use environment variable
MODEL_PATH=/path/to/your/model python app.py
```

5. **Access the interface**:
Open your browser and navigate to `http://localhost:7860`

## Usage

### Speech-to-Speech Chat

1. Upload audio or record a message using your microphone
2. Optionally add text input alongside audio
3. (Optional) Customize the system prompt for specific behaviors
4. Click **"Send"** to get a response with both text and audio
5. Continue the conversation - your chat history is preserved

**Example system prompts:**
- `Respond with interleaved text and audio.` (default)
- `Respond only with audio.`
- `Respond only with text.`

### Automatic Speech Recognition (ASR)

1. Upload an audio file or record speech using your microphone
2. Click **"Transcribe"**
3. View the transcribed text

### Text-to-Speech (TTS)

1. Enter text in the input field
2. Select a voice:
   - **US Male** / **US Female**
   - **UK Male** / **UK Female**
3. Click **"Synthesize"**
4. Listen to the generated audio

## Model Information

- **Model**: LFM2.5-Audio-1.5B
- **Provider**: Liquid AI
- **Repository**: [Hugging Face - LiquidAI/LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B)
- **License**: LFM Open License v1.0

## Local Model Configuration

The app now supports loading models from a local directory instead of downloading from Hugging Face each time.

📖 **For detailed setup instructions, see [LOCAL_MODEL_SETUP.md](LOCAL_MODEL_SETUP.md)**

### Model Path Options (in order of precedence):

1. **Command-line argument**: `--model-path /path/to/model`
2. **Environment variable**: `MODEL_PATH=/path/to/model`
3. **Default path**: `./models/LFM2.5-Audio-1.5B`

### Setting Up Local Models:

To use local models, you can:

1. **Download from Hugging Face CLI**:
   ```bash
   # Install huggingface-hub if not already installed
   pip install huggingface-hub
   
   # Download the model
   huggingface-cli download LiquidAI/LFM2.5-Audio-1.5B --local-dir ./models/LFM2.5-Audio-1.5B
   ```

2. **Use git-lfs**:
   ```bash
   git lfs install
   git clone https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B ./models/LFM2.5-Audio-1.5B
   ```

3. **Manual download**: Download model files from the Hugging Face repository and place them in your chosen directory

### Fallback Behavior:

If the specified local model path doesn't exist, the app will automatically fall back to downloading the model from Hugging Face. This ensures the app always works, even if local models aren't set up yet.

## Technical Details

### Audio Processing
- **Sample Rate**: 24,000 Hz
- **Audio Format**: WAV
- **Generation Parameters**:
  - Text temperature: 1.0 (default)
  - Audio temperature: 1.0 (speech-to-speech), 0.8 (TTS)
  - Max tokens: 512
  - Audio top-k: 4 (speech-to-speech), 64 (TTS)

### Architecture
The interface uses Gradio with PyTorch and torchaudio to load and run the LFM2.5-Audio-1.5B model. The model can be loaded from a local directory or automatically downloaded from Hugging Face if not found locally.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for faster inference)
- torchaudio
- Gradio 4.0+
- liquid-audio library

## Notes

- By default, models are loaded from `./models/LFM2.5-Audio-1.5B` directory
- If local models are not found, they will be automatically downloaded from Hugging Face
- First model load may take longer if downloading from Hugging Face
- GPU is strongly recommended for real-time performance
- The model supports multi-turn conversations with full history preservation
- Audio files are temporarily stored during processing and cleaned up afterward

## License

This project is licensed under the LFM Open License v1.0. See the original model license for more details.

## References

- [Liquid AI on Hugging Face](https://huggingface.co/LiquidAI)
- [LFM2.5-Audio-1.5B Model](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B)

## Contact

For questions about the Liquid Audio model, visit the Liquid AI Hugging Face repository.

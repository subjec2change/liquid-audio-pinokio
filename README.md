# Liquid Audio

🎙️ **Liquid Audio** — A Gradio web interface powered by a locally running
[llama-server](https://github.com/ggerganov/llama.cpp) (llama.cpp) endpoint.
Runs entirely on-device with CUDA-accelerated GGUF models — no cloud API required.

## Overview

The app exposes three tabs via a Gradio web UI:

| Tab | Description |
|-----|-------------|
| 💬 Speech-to-Speech Chat | Multi-turn text/audio conversation via llama-server |
| 📝 Automatic Speech Recognition | Send audio context to llama-server for transcription/response |
| 🔊 Text-to-Speech | Generate text responses styled by voice profile |

---

## Linux / CUDA Setup

### Prerequisites

- Linux (x86_64)
- NVIDIA GPU with CUDA 12.x drivers installed
- `cmake` ≥ 3.21, `gcc`/`g++` ≥ 11, `git`
- Python 3.10+

---

### Step 1 — Build llama.cpp with CUDA

```bash
git clone https://github.com/ggerganov/llama.cpp
cmake -S llama.cpp -B llama.cpp/build \
      -DGGML_CUDA=ON \
      -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j$(nproc)
```

> **Tip:** add `llama.cpp/build/bin` to your `PATH` so `llama-server` is
> available system-wide:
> ```bash
> export PATH="$PWD/llama.cpp/build/bin:$PATH"
> ```

---

### Step 2 — Obtain a GGUF model

Download any GGUF model from Hugging Face, for example:

```bash
# Install huggingface-hub if needed
pip install huggingface-hub

# Example: Mistral 7B Q4_K_M (≈ 4 GB)
huggingface-cli download \
    bartowski/Mistral-7B-Instruct-v0.3-GGUF \
    Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
    --local-dir ~/models
```

Any instruction-tuned GGUF model works.  For audio transcription, use a
multimodal model or a whisper GGUF served via llama-server.

---

### Step 3 — Start llama-server

Use the provided helper script:

```bash
bash scripts/run-llama-server.sh \
    --model ~/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
```

Or set environment variables and run directly:

```bash
export LLAMA_MODEL=~/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
export LLAMA_HOST=127.0.0.1
export LLAMA_PORT=8080
export LLAMA_N_GPU_LAYERS=99   # offload all layers to GPU
bash scripts/run-llama-server.sh
```

The script automatically searches for `llama-server` in `PATH` and in
`./llama.cpp/build/bin/`.  Once running you should see:

```
llama server listening at http://127.0.0.1:8080
```

---

### Step 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5 — Start the Python app

```bash
python app.py --no-share
```

Open your browser at `http://localhost:7860`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_BASE_URL` | `http://127.0.0.1:8080` | Base URL of the running llama-server |
| `LLAMA_MODEL` | `local-model` | Model identifier sent in API requests (informational) |
| `LLAMA_API_KEY` | `not-needed` | API key (dummy value; required by the OpenAI client) |
| `LLAMA_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLAMA_MAX_TOKENS` | `512` | Maximum tokens to generate per request |

Example:

```bash
LLAMA_BASE_URL=http://127.0.0.1:8080 \
LLAMA_MODEL=mistral-7b \
LLAMA_TEMPERATURE=0.5 \
python app.py --no-share
```

---

## Usage

### Speech-to-Speech Chat

1. Upload audio or record a message using your microphone
2. Optionally add text input alongside audio
3. (Optional) Customise the system prompt for specific behaviours
4. Click **Send** to get a streamed text response from llama-server
5. Continue the conversation — your chat history is preserved

### Automatic Speech Recognition (ASR)

1. Upload an audio file or record speech
2. Click **Transcribe**
3. View the response from llama-server

> For true audio transcription, use a multimodal GGUF (e.g., whisper.cpp
> served via llama-server) and pass the audio file path as context.

### Text-to-Speech (TTS)

1. Enter text
2. Select a voice style (US/UK, Male/Female)
3. Click **Generate** — llama-server returns a styled text response
4. Pipe the output to a TTS engine (e.g., `espeak`, `piper`) for audio if required

---

## Architecture

```
Browser → Gradio (port 7860) → app.py
                                  │
                        OpenAI-compatible API
                                  │
                         llama-server (port 8080)
                                  │
                         GGUF model (CUDA / NVIDIA GPU)
```

`app.py` uses the `openai` Python package with a custom `base_url` to talk to
the OpenAI-compatible `/v1/chat/completions` endpoint exposed by llama-server.

---

## scripts/run-llama-server.sh flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Path to `.gguf` model file |
| `--host` | `127.0.0.1` | Host to bind to |
| `--port` | `8080` | Port to listen on |
| `--n-gpu-layers` | `99` | Number of model layers to offload to GPU |
| `--ctx-size` | `4096` | Context size in tokens |
| `--threads` | `4` | CPU threads for non-GPU operations |

---

## Troubleshooting

### "Cannot connect to llama-server"

Make sure `llama-server` is running and listening on the configured
`LLAMA_BASE_URL` before starting the Python app.

### CUDA out-of-memory

Reduce `--n-gpu-layers` (e.g. `--n-gpu-layers 20`) to keep some layers on CPU.

### llama-server not found

Ensure `llama.cpp/build/bin` is in your `PATH`, or run the helper script from
the repository root directory containing the `llama.cpp/` folder.

---

## Requirements

- Python 3.10+
- `gradio>=5.50.0`
- `openai>=1.0.0`
- llama.cpp built with `GGML_CUDA=ON`
- NVIDIA GPU with CUDA drivers

## License

See the original model licenses for any GGUF models you download.


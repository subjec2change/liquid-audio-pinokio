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
- `ffmpeg` installed on the system (required by OpenAI Whisper for audio decoding)

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

### Single-model configuration (backward compatible)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_BASE_URL` | `http://127.0.0.1:8080` | Base URL of the running llama-server |
| `LLAMA_MODEL` | `local-model` | Model identifier sent in API requests |
| `LLAMA_API_KEY` | `not-needed` | API key (dummy value; required by the OpenAI client) |
| `LLAMA_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLAMA_MAX_TOKENS` | `512` | Maximum tokens to generate per request |

### ASR configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `base` | OpenAI Whisper model size for ASR (`tiny`, `base`, `small`, `medium`, `large`) |

### Multi-model configuration

| Variable | Format | Description |
|----------|--------|-------------|
| `LLAMA_MODELS` | JSON `{"alias": "http://host:port", ...}` | Map of model names → server base URLs. **Takes precedence over `LLAMA_BASE_URL`/`LLAMA_MODEL` when set.** |

Example (single model, backward-compatible):

```bash
LLAMA_BASE_URL=http://127.0.0.1:8080 \
LLAMA_MODEL=mistral-7b \
LLAMA_TEMPERATURE=0.5 \
python app.py --no-share
```

Example (multiple models):

```bash
export LLAMA_MODELS='{"mistral-7b": "http://127.0.0.1:8080",
                      "llama-3-8b": "http://127.0.0.1:8081"}'
python app.py --no-share
```

---

## Running Multiple Models Simultaneously

Each model runs in its own `llama-server` process on a different port.
The Python app routes each request to the selected model's server.

### Step 1 — Start all servers with the multi-model script

```bash
bash scripts/run-llama-server-multi.sh \
    --model "mistral-7b:~/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf" \
    --model "llama-3-8b:~/models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
```

The script assigns consecutive ports starting at `8080` and prints the
`LLAMA_MODELS` export string to copy-paste:

```
export LLAMA_MODELS='{"mistral-7b":"http://127.0.0.1:8080","llama-3-8b":"http://127.0.0.1:8081"}'
python app.py --no-share
```

### Step 2 — Export `LLAMA_MODELS` and start the app

```bash
export LLAMA_MODELS='{"mistral-7b":"http://127.0.0.1:8080","llama-3-8b":"http://127.0.0.1:8081"}'
python app.py --no-share
```

The app's model selector dropdown (present on every tab) lets you switch
between models without restarting anything.

### GPU memory considerations

Running multiple large models at the same time requires enough VRAM for all
of them simultaneously.  Strategies to reduce VRAM usage:

- Use smaller quantised models (Q4_K_M, Q3_K_M)
- Reduce `--n-gpu-layers` so some layers run on CPU
- Use `--ctx-size 2048` (or lower) to shrink the KV-cache per instance

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
3. The audio is transcribed locally using OpenAI Whisper (no llama-server required for this tab)

### Text-to-Speech (TTS)

1. Enter text
2. Select a voice style (US/UK, Male/Female)
3. Click **Generate** — llama-server returns a styled text response
4. Pipe the output to a TTS engine (e.g., `espeak`, `piper`) for audio if required

---

## Architecture

### Single model

```
Browser → Gradio (port 7860) → app.py
                                  │  (model selector = "mistral-7b")
                        OpenAI-compatible API
                                  │
                         llama-server :8080
                                  │
                         mistral-7b.gguf  (CUDA)
```

### Multiple models

```
Browser → Gradio (port 7860) → app.py
                                  ├── model selector = "mistral-7b"
                                  │       └── llama-server :8080 → mistral-7b.gguf
                                  └── model selector = "llama-3-8b"
                                          └── llama-server :8081 → llama-3-8b.gguf
```

`app.py` uses the `openai` Python package with a custom `base_url` (looked up
from `LLAMA_MODELS`) to talk to the OpenAI-compatible `/v1/chat/completions`
endpoint exposed by the selected llama-server instance.

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

## scripts/run-llama-server-multi.sh flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(one or more required)* | `alias:/path/to/model.gguf` — repeat for each model |
| `--base-port` | `8080` | First port; each subsequent model uses `base-port + n` |
| `--host` | `127.0.0.1` | Host to bind all servers to |
| `--n-gpu-layers` | `99` | GPU layers (applied to all instances) |
| `--ctx-size` | `4096` | Context size (applied to all instances) |
| `--threads` | `4` | CPU threads (applied to all instances) |

---

## Troubleshooting

### "Cannot connect to llama-server"

Make sure `llama-server` is running and listening on the URL configured for
the selected model.  Use `LLAMA_MODELS` (or `LLAMA_BASE_URL`) to point the
app at the correct address.

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
- `openai-whisper` (for local ASR transcription)
- `ffmpeg` installed on the system (required by OpenAI Whisper)
- llama.cpp built with `GGML_CUDA=ON`
- NVIDIA GPU with CUDA drivers

## License

See the original model licenses for any GGUF models you download.


#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run-llama-server.sh — Start llama-server (llama.cpp) with CUDA on Linux
#
# Usage:
#   bash scripts/run-llama-server.sh [--model /path/to/model.gguf] [options]
#
# Environment variable overrides (take precedence over defaults below):
#   LLAMA_MODEL        Path to the .gguf model file
#   LLAMA_HOST         Host to bind to          (default: 127.0.0.1)
#   LLAMA_PORT         Port to listen on         (default: 8080)
#   LLAMA_N_GPU_LAYERS Number of layers on GPU   (default: 99 = all)
#   LLAMA_CTX_SIZE     Context size in tokens    (default: 4096)
#   LLAMA_THREADS      CPU threads for non-GPU   (default: 4)
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------- defaults (override via env or --flags below) -------------------
MODEL="${LLAMA_MODEL:-}"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8080}"
N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-99}"
CTX_SIZE="${LLAMA_CTX_SIZE:-4096}"
THREADS="${LLAMA_THREADS:-4}"

# ---------- parse CLI flags ------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2";        shift 2 ;;
        --host)           HOST="$2";         shift 2 ;;
        --port)           PORT="$2";         shift 2 ;;
        --n-gpu-layers)   N_GPU_LAYERS="$2"; shift 2 ;;
        --ctx-size)       CTX_SIZE="$2";     shift 2 ;;
        --threads)        THREADS="$2";      shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------- validate -------------------------------------------------------
if [[ -z "$MODEL" ]]; then
    echo "Error: no model specified."
    echo "  Set LLAMA_MODEL env var, or pass --model /path/to/model.gguf"
    echo ""
    echo "Example:"
    echo "  bash scripts/run-llama-server.sh --model ~/models/mistral-7b-q4_k_m.gguf"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: model file not found: $MODEL"
    exit 1
fi

# ---------- locate llama-server --------------------------------------------
if command -v llama-server &>/dev/null; then
    LLAMA_SERVER_BIN="llama-server"
elif [[ -x "./llama.cpp/build/bin/llama-server" ]]; then
    LLAMA_SERVER_BIN="./llama.cpp/build/bin/llama-server"
elif [[ -x "./build/bin/llama-server" ]]; then
    LLAMA_SERVER_BIN="./build/bin/llama-server"
else
    echo "Error: llama-server not found in PATH or common build directories."
    echo ""
    echo "Build llama.cpp with CUDA support:"
    echo "  git clone https://github.com/ggerganov/llama.cpp"
    echo "  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON"
    echo "  cmake --build llama.cpp/build --config Release -j\$(nproc)"
    echo ""
    echo "Then add llama.cpp/build/bin to your PATH, or re-run this script"
    echo "from the repo root containing the llama.cpp/ directory."
    exit 1
fi

# ---------- launch ---------------------------------------------------------
echo "Starting llama-server"
echo "  binary   : $LLAMA_SERVER_BIN"
echo "  model    : $MODEL"
echo "  host     : $HOST"
echo "  port     : $PORT"
echo "  gpu layers: $N_GPU_LAYERS"
echo "  ctx size : $CTX_SIZE"
echo ""
echo "OpenAI-compatible endpoint: http://${HOST}:${PORT}/v1"
echo "Press Ctrl+C to stop."
echo ""

exec "$LLAMA_SERVER_BIN" \
    --model       "$MODEL" \
    --host        "$HOST" \
    --port        "$PORT" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --ctx-size    "$CTX_SIZE" \
    --threads     "$THREADS"

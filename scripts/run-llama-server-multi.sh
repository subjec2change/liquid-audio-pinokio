#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run-llama-server-multi.sh — Launch one llama-server instance per model
#                              (llama.cpp, CUDA) on consecutive ports
#
# Usage:
#   bash scripts/run-llama-server-multi.sh \
#       --model "mistral-7b:/path/to/mistral-7b-q4_k_m.gguf" \
#       --model "llama-3-8b:/path/to/llama-3-8b-q4_k_m.gguf" \
#       [--base-port 8080] [--host 127.0.0.1]
#       [--n-gpu-layers 99] [--ctx-size 4096] [--threads 4]
#
# Each --model value is  <alias>:<path>  where:
#   alias  = short name used in the UI / LLAMA_MODELS JSON
#   path   = path to the .gguf model file
#
# The script prints the LLAMA_MODELS JSON to export before starting the app:
#   export LLAMA_MODELS='{"mistral-7b":"http://127.0.0.1:8080","llama-3-8b":"http://127.0.0.1:8081"}'
#   python app.py --no-share
#
# Environment variable overrides:
#   LLAMA_HOST         Host to bind all servers to (default: 127.0.0.1)
#   LLAMA_BASE_PORT    First port to use           (default: 8080)
#   LLAMA_N_GPU_LAYERS Layers offloaded to GPU     (default: 99 = all)
#   LLAMA_CTX_SIZE     Context size in tokens      (default: 4096)
#   LLAMA_THREADS      CPU threads for non-GPU     (default: 4)
# ---------------------------------------------------------------------------
set -euo pipefail

HOST="${LLAMA_HOST:-127.0.0.1}"
BASE_PORT="${LLAMA_BASE_PORT:-8080}"
N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-99}"
CTX_SIZE="${LLAMA_CTX_SIZE:-4096}"
THREADS="${LLAMA_THREADS:-4}"

# Parallel arrays: model aliases and their file paths
declare -a ALIASES=()
declare -a PATHS=()

# ---------- parse CLI flags ------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            # Expected format:  alias:/path/to/model.gguf
            RAW="$2"; shift 2
            ALIAS="${RAW%%:*}"
            PATH_="${RAW#*:}"
            if [[ -z "$ALIAS" || "$ALIAS" == "$RAW" ]]; then
                echo "Error: --model value must be in the form  alias:/path/to/model.gguf"
                echo "  Got: $RAW"
                exit 1
            fi
            ALIASES+=("$ALIAS")
            PATHS+=("$PATH_")
            ;;
        --base-port)    BASE_PORT="$2";    shift 2 ;;
        --host)         HOST="$2";         shift 2 ;;
        --n-gpu-layers) N_GPU_LAYERS="$2"; shift 2 ;;
        --ctx-size)     CTX_SIZE="$2";     shift 2 ;;
        --threads)      THREADS="$2";      shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ${#ALIASES[@]} -eq 0 ]]; then
    echo "Error: no models specified."
    echo ""
    echo "Example:"
    echo "  bash scripts/run-llama-server-multi.sh \\"
    echo "      --model \"mistral-7b:~/models/mistral-7b-q4_k_m.gguf\" \\"
    echo "      --model \"llama-3-8b:~/models/llama-3-8b-q4_k_m.gguf\""
    exit 1
fi

# ---------- validate model paths -------------------------------------------
for i in "${!PATHS[@]}"; do
    # Expand ~ manually since it is not expanded inside quoted strings
    # (handles ~/path but not ~username/path which is an uncommon edge case)
    MODEL_PATH="${PATHS[$i]}"
    MODEL_PATH="${MODEL_PATH/#\~/$HOME}"
    PATHS[$i]="$MODEL_PATH"
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "Error: model file not found for '${ALIASES[$i]}': $MODEL_PATH"
        exit 1
    fi
done

# ---------- locate llama-server binary -------------------------------------
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
    exit 1
fi

# ---------- launch servers -------------------------------------------------
declare -a PIDS=()
declare -a PORTS=()

cleanup() {
    echo ""
    echo "Shutting down llama-server instances..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

echo "Starting ${#ALIASES[@]} llama-server instance(s)..."
echo ""

for i in "${!ALIASES[@]}"; do
    PORT=$(( BASE_PORT + i ))
    ALIAS="${ALIASES[$i]}"
    MODEL_PATH="${PATHS[$i]}"

    echo "  [$((i+1))/${#ALIASES[@]}] $ALIAS"
    echo "       model : $MODEL_PATH"
    echo "       listen: http://${HOST}:${PORT}/v1"

    "$LLAMA_SERVER_BIN" \
        --model        "$MODEL_PATH" \
        --host         "$HOST" \
        --port         "$PORT" \
        --n-gpu-layers "$N_GPU_LAYERS" \
        --ctx-size     "$CTX_SIZE" \
        --threads      "$THREADS" \
        &>/tmp/llama-server-"$ALIAS".log &

    PIDS+=($!)
    PORTS+=("$PORT")
done

echo ""

# ---------- print LLAMA_MODELS export string --------------------------------
JSON="{"
for i in "${!ALIASES[@]}"; do
    PORT="${PORTS[$i]}"
    ALIAS="${ALIASES[$i]}"
    [[ $i -gt 0 ]] && JSON+=","
    JSON+="\"$ALIAS\":\"http://${HOST}:${PORT}\""
done
JSON+="}"

echo "All servers launched.  Export LLAMA_MODELS before starting the app:"
echo ""
echo "  export LLAMA_MODELS='${JSON}'"
echo "  python app.py --no-share"
echo ""
echo "Logs: /tmp/llama-server-<alias>.log"
echo "Press Ctrl+C to stop all servers."
echo ""

# ---------- wait for any server to exit ------------------------------------
# Wait for any server to exit.  `wait -n` (Bash 4.3+) returns as soon as one
# child exits; plain `wait` is the fallback for older shells (waits for all).
wait -n 2>/dev/null || wait

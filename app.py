import gradio as gr
from openai import OpenAI, APIConnectionError
import os
import json
import argparse
import threading
import torch
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# llama-server connection configuration (override with environment variables)
# ---------------------------------------------------------------------------
#
# Multi-model mode (recommended):
#   Set LLAMA_MODELS to a JSON object mapping model names to their base URLs:
#     LLAMA_MODELS='{"mistral-7b": "http://127.0.0.1:8080",
#                    "llama-3-8b": "http://127.0.0.1:8081"}'
#
# Single-model mode (backward compatible):
#   If LLAMA_MODELS is not set, the app falls back to LLAMA_BASE_URL /
#   LLAMA_MODEL, identical to the previous behaviour.
# ---------------------------------------------------------------------------

LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY", "not-needed")
LLAMA_TEMPERATURE = float(os.environ.get("LLAMA_TEMPERATURE", "0.7"))
LLAMA_MAX_TOKENS = int(os.environ.get("LLAMA_MAX_TOKENS", "512"))

# Build the models dict: {name -> base_url}
_models_json = os.environ.get("LLAMA_MODELS", "")
if _models_json:
    try:
        LLAMA_MODELS: dict[str, str] = json.loads(_models_json)
        if not isinstance(LLAMA_MODELS, dict) or not LLAMA_MODELS:
            raise ValueError("LLAMA_MODELS must be a non-empty JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(
            f"Error: LLAMA_MODELS is not valid JSON.\n"
            f"  Expected format: '{{\"model-name\": \"http://host:port\", ...}}'\n"
            f"  Got: {_models_json!r}\n"
            f"  Detail: {exc}"
        ) from exc
else:
    # Backward-compatible single-model fallback
    _base_url = os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8080")
    _model_name = os.environ.get("LLAMA_MODEL", "local-model")
    LLAMA_MODELS = {_model_name: _base_url}

MODEL_NAMES: list[str] = list(LLAMA_MODELS.keys())
DEFAULT_MODEL: str = MODEL_NAMES[0]

# ---------------------------------------------------------------------------
# faster-whisper configuration (model is loaded lazily on first use)
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
_whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
_whisper_compute_type = "float16" if _whisper_device == "cuda" else "int8"
_whisper_model: WhisperModel | None = None
_whisper_model_lock = threading.Lock()


def _get_whisper_model() -> WhisperModel:
    """Return the faster-whisper model, loading it on first call (thread-safe)."""
    global _whisper_model
    if _whisper_model is None:
        with _whisper_model_lock:
            if _whisper_model is None:
                try:
                    _whisper_model = WhisperModel(
                        WHISPER_MODEL_SIZE, device=_whisper_device, compute_type=_whisper_compute_type
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load faster-whisper model '{WHISPER_MODEL_SIZE}'. "
                        f"Check your network connection or set WHISPER_MODEL_SIZE to a valid size "
                        f"(tiny, base, small, medium, large-v3). Detail: {exc}"
                    ) from exc
    return _whisper_model


_SERVER_UNAVAILABLE_MSG = (
    "⚠️ Cannot connect to llama-server at {url}.\n"
    "Please start llama-server first:\n"
    "  bash scripts/run-llama-server.sh  (single model)\n"
    "  bash scripts/run-llama-server-multi.sh  (multiple models)\n"
    "See README.md for full setup instructions."
)


def get_client(model_name: str) -> tuple[OpenAI, str]:
    """Return an (OpenAI client, model name) pair for the given model alias."""
    base_url = LLAMA_MODELS[model_name]
    client = OpenAI(
        base_url=f"{base_url}/v1",
        api_key=LLAMA_API_KEY,
    )
    return client, model_name


def _server_error_msg(model_name: str) -> str:
    url = LLAMA_MODELS.get(model_name, f"<unknown model '{model_name}'>")
    return _SERVER_UNAVAILABLE_MSG.format(url=url)


def speech_to_speech_chat(audio_input, text_input, chat_history, system_prompt, model_name):
    """Handle multi-turn conversation via the selected llama-server."""
    try:
        client, model = get_client(model_name)

        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."}
        ]

        # Rebuild message history (stored as list of (role, text) pairs)
        for role, text_data in chat_history:
            messages.append({"role": role, "content": text_data})

        user_message = text_input or ""
        if audio_input is not None:
            # Note: llama-server cannot process raw audio without a multimodal
            # GGUF model (e.g. whisper-based).  The tag below is a text-only
            # placeholder so the model is at least aware audio was provided.
            user_message = f"[Audio input received] {user_message}".strip()

        if not user_message:
            yield "Please provide text or audio input.", chat_history
            return

        messages.append({"role": "user", "content": user_message})

        full_response = ""
        with client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=LLAMA_TEMPERATURE,
            max_tokens=LLAMA_MAX_TOKENS,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    yield full_response, chat_history

        new_history = chat_history + [
            ("user", user_message),
            ("assistant", full_response),
        ]
        yield full_response, new_history

    except APIConnectionError:
        yield _server_error_msg(model_name), chat_history
    except Exception as e:
        yield f"Error: {str(e)}", chat_history


def asr_transcription(audio_input, model_name):
    """Transcribe audio using faster-whisper locally."""
    try:
        if audio_input is None:
            yield "Please provide an audio input."
            return

        segments, info = _get_whisper_model().transcribe(audio_input, beam_size=5)
        transcript = " ".join(segment.text for segment in segments)

        if not transcript.strip():
            yield "No speech detected in the audio."
        else:
            yield transcript.strip()

    except Exception as e:
        yield f"Transcription error: {str(e)}"


def tts_synthesis(text_input, voice_selection, model_name):
    """Generate text via the selected llama-server (audio output requires a separate TTS backend)."""
    try:
        if not text_input:
            yield "Please provide text input."
            return

        client, model = get_client(model_name)

        voice_styles = {
            "US Male": "Respond in a clear, professional US English style.",
            "US Female": "Respond in a warm, friendly US English style.",
            "UK Male": "Respond in a formal, British English style.",
            "UK Female": "Respond in a polished, British English style.",
        }

        style = voice_styles.get(voice_selection, voice_styles["US Male"])

        messages = [
            {"role": "system", "content": f"You are a helpful assistant. {style}"},
            {"role": "user", "content": text_input},
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=LLAMA_TEMPERATURE,
            max_tokens=LLAMA_MAX_TOKENS,
        )
        yield response.choices[0].message.content

    except APIConnectionError:
        yield _server_error_msg(model_name)
    except Exception as e:
        yield f"Error: {str(e)}"

def _model_selector(tab_id: str) -> gr.Dropdown:
    """Return a pre-configured model-selector dropdown for the given tab."""
    return gr.Dropdown(
        choices=MODEL_NAMES,
        value=DEFAULT_MODEL,
        label="Model",
        info="Select which llama-server instance to query.",
        interactive=True,
        elem_id=f"model_selector_{tab_id}",
    )


def create_ui():
    """Create the Gradio interface"""

    models_summary = ", ".join(
        f"`{name}` → {url}" for name, url in LLAMA_MODELS.items()
    )

    with gr.Blocks(title="Liquid Audio UI") as demo:
        gr.Markdown("# 🎙️ Liquid Audio — llama-server backend")
        gr.Markdown(
            "Speech-to-Speech, ASR, and TTS via a local "
            "[llama-server](https://github.com/ggerganov/llama.cpp) endpoint"
        )

        with gr.Tabs():
            # Speech-to-Speech Chat Tab
            with gr.Tab("💬 Speech-to-Speech Chat"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Input")
                        chat_model = _model_selector("chat")
                        audio_input = gr.Audio(
                            label="Upload Audio (or record)",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        text_input = gr.Textbox(
                            label="Text Input (optional)",
                            placeholder="Or type a message...",
                            lines=2,
                        )
                        system_prompt = gr.Textbox(
                            label="System Prompt (optional)",
                            placeholder="Leave blank for default: 'You are a helpful assistant.'",
                            lines=2,
                        )
                        chat_submit = gr.Button("Send", variant="primary", size="lg")

                    with gr.Column():
                        gr.Markdown("### Response")
                        text_output = gr.Textbox(
                            label="Text Response",
                            interactive=False,
                            lines=5,
                        )

                chat_history = gr.State([])

                chat_submit.click(
                    fn=speech_to_speech_chat,
                    inputs=[audio_input, text_input, chat_history, system_prompt, chat_model],
                    outputs=[text_output, chat_history],
                )

                gr.Markdown("---")
                gr.Markdown(
                    "**How to use:**\n"
                    "1. Select a model from the dropdown\n"
                    "2. Upload audio or record a message\n"
                    "3. Optionally add text input\n"
                    "4. Click Send to get a response from llama-server\n"
                    "5. Continue the conversation — your history is preserved"
                )

            # ASR Tab
            with gr.Tab("📝 Automatic Speech Recognition"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Audio Input")
                        asr_model = _model_selector("asr")
                        asr_audio_input = gr.Audio(
                            label="Upload Audio or Record",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        asr_submit = gr.Button("Transcribe", variant="primary", size="lg")

                    with gr.Column():
                        gr.Markdown("### Transcription")
                        asr_output = gr.Textbox(
                            label="Transcribed Text",
                            interactive=False,
                            lines=10,
                        )

                asr_submit.click(
                    fn=asr_transcription,
                    inputs=[asr_audio_input, asr_model],
                    outputs=asr_output,
                )

                gr.Markdown("---")
                gr.Markdown(
                    "**How to use:**\n"
                    "1. Upload an audio file or record speech\n"
                    "2. Click Transcribe\n"
                    "3. View the transcription — powered locally by faster-whisper "
                    f"(model: `{WHISPER_MODEL_SIZE}`, device: `{_whisper_device}`)\n"
                    "4. Set the `WHISPER_MODEL_SIZE` environment variable to change "
                    "the model size (`tiny`, `base`, `small`, `medium`, `large-v3`)"
                )

            # TTS Tab
            with gr.Tab("🔊 Text-to-Speech"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Text Input")
                        tts_model = _model_selector("tts")
                        tts_text_input = gr.Textbox(
                            label="Enter Text to Convert",
                            placeholder="Type the text you want to synthesize...",
                            lines=5,
                        )
                        voice_selection = gr.Radio(
                            choices=["US Male", "US Female", "UK Male", "UK Female"],
                            value="US Male",
                            label="Select Voice Style",
                        )
                        tts_submit = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column():
                        gr.Markdown("### Generated Text")
                        tts_output = gr.Textbox(
                            label="Generated Response",
                            interactive=False,
                            lines=10,
                        )

                tts_submit.click(
                    fn=tts_synthesis,
                    inputs=[tts_text_input, voice_selection, tts_model],
                    outputs=tts_output,
                )

                gr.Markdown("---")
                gr.Markdown(
                    "**How to use:**\n"
                    "1. Select a model from the dropdown\n"
                    "2. Enter text in the input field\n"
                    "3. Select a voice style\n"
                    "4. Click Generate — llama-server returns a text response; "
                    "pipe the output to a TTS engine for audio if required"
                )

        gr.Markdown("---")
        gr.Markdown(
            f"**Configured models:** {models_summary}\n\n"
            "See [README.md](README.md) for Linux/CUDA setup and multi-model instructions."
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Liquid Audio — llama-server backend"
    )
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="Run locally without creating a public shareable link (default: share is enabled).",
    )
    args = parser.parse_args()

    print("Configured models:")
    for name, url in LLAMA_MODELS.items():
        print(f"  {name:30s} → {url}")

    demo = create_ui()
    demo.launch(share=not args.no_share)

import gradio as gr
from openai import OpenAI, APIConnectionError
import os
import argparse

# ---------------------------------------------------------------------------
# llama-server connection configuration (override with environment variables)
# ---------------------------------------------------------------------------
LLAMA_BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8080")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "local-model")
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY", "not-needed")
LLAMA_TEMPERATURE = float(os.environ.get("LLAMA_TEMPERATURE", "0.7"))
LLAMA_MAX_TOKENS = int(os.environ.get("LLAMA_MAX_TOKENS", "512"))

_SERVER_UNAVAILABLE_MSG = (
    "⚠️ Cannot connect to llama-server at {url}.\n"
    "Please start llama-server first:\n"
    "  bash scripts/run-llama-server.sh\n"
    "See README.md for full setup instructions."
)


def get_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at the local llama-server."""
    return OpenAI(
        base_url=f"{LLAMA_BASE_URL}/v1",
        api_key=LLAMA_API_KEY,
    )


def _server_error_msg() -> str:
    return _SERVER_UNAVAILABLE_MSG.format(url=LLAMA_BASE_URL)


def speech_to_speech_chat(audio_input, text_input, chat_history, system_prompt):
    """Handle multi-turn conversation via llama-server."""
    try:
        client = get_client()

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
            model=LLAMA_MODEL,
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
        yield _server_error_msg(), chat_history
    except Exception as e:
        yield f"Error: {str(e)}", chat_history


def asr_transcription(audio_input):
    """Transcribe audio via llama-server."""
    try:
        if audio_input is None:
            yield "Please provide an audio input."
            return

        client = get_client()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "When the user mentions an audio file, acknowledge it and note "
                    "that actual audio transcription requires a whisper.cpp endpoint "
                    "or a multimodal GGUF model."
                ),
            },
            {
                "role": "user",
                "content": (
                    "An audio file was uploaded. "
                    "For full audio transcription, connect a whisper.cpp endpoint "
                    "or a multimodal GGUF model to llama-server."
                ),
            },
        ]

        response = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=messages,
            temperature=LLAMA_TEMPERATURE,
            max_tokens=LLAMA_MAX_TOKENS,
        )
        yield response.choices[0].message.content

    except APIConnectionError:
        yield _server_error_msg()
    except Exception as e:
        yield f"Error: {str(e)}"


def tts_synthesis(text_input, voice_selection):
    """Generate text via llama-server (audio output requires a separate TTS backend)."""
    try:
        if not text_input:
            yield "Please provide text input."
            return

        client = get_client()

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
            model=LLAMA_MODEL,
            messages=messages,
            temperature=LLAMA_TEMPERATURE,
            max_tokens=LLAMA_MAX_TOKENS,
        )
        yield response.choices[0].message.content

    except APIConnectionError:
        yield _server_error_msg()
    except Exception as e:
        yield f"Error: {str(e)}"

def create_ui():
    """Create the Gradio interface"""

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
                    inputs=[audio_input, text_input, chat_history, system_prompt],
                    outputs=[text_output, chat_history],
                )

                gr.Markdown("---")
                gr.Markdown(
                    "**How to use:**\n"
                    "1. Upload audio or record a message\n"
                    "2. Optionally add text input\n"
                    "3. Click Send to get a response from llama-server\n"
                    "4. Continue the conversation — your history is preserved"
                )

            # ASR Tab
            with gr.Tab("📝 Automatic Speech Recognition"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Audio Input")
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
                    inputs=asr_audio_input,
                    outputs=asr_output,
                )

                gr.Markdown("---")
                gr.Markdown(
                    "**How to use:**\n"
                    "1. Upload an audio file or record speech\n"
                    "2. Click Transcribe — for full audio transcription use a "
                    "multimodal or whisper GGUF model with llama-server"
                )

            # TTS Tab
            with gr.Tab("🔊 Text-to-Speech"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Text Input")
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
                    inputs=[tts_text_input, voice_selection],
                    outputs=tts_output,
                )

                gr.Markdown("---")
                gr.Markdown(
                    "**How to use:**\n"
                    "1. Enter text in the input field\n"
                    "2. Select a voice style\n"
                    "3. Click Generate — llama-server returns a text response; "
                    "pipe the output to a TTS engine for audio if required"
                )

        gr.Markdown("---")
        gr.Markdown(
            f"**Backend:** llama-server at `{LLAMA_BASE_URL}` | "
            "**Model:** set via `LLAMA_MODEL` env var\n"
            "See [README.md](README.md) for full Linux/CUDA setup instructions."
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

    print(f"Connecting to llama-server at: {LLAMA_BASE_URL}")
    print(f"Model identifier          : {LLAMA_MODEL}")

    demo = create_ui()
    demo.launch(share=not args.no_share)

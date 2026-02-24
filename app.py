import argparse
import os
import threading

import gradio as gr
import torch
import torchaudio
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Police Report Cleanup Prompt
# ---------------------------------------------------------------------------

POLICE_REPORT_CLEANUP_PROMPT = """\
You are a professional police-report transcription editor. Your ONLY job is to \
clean up the raw speech-to-text transcript provided by the user. You must:

1. Fix all capitalization — proper nouns, start of sentences, acronyms (e.g., DUI, \
DOB, VIN, ID, SSN, EMS, DOA, APB, BOLO), titles (Officer, Sergeant, Detective), \
and legal/agency names.
2. Add correct punctuation — periods, commas, colons, semicolons, and quotation \
marks where appropriate.
3. Fix grammar — subject-verb agreement, tense consistency, articles, and \
prepositions.
4. Preserve all factual content EXACTLY — do NOT add, remove, rephrase, \
summarize, or editorialize any information. Do NOT infer or fill in missing \
details.
5. Structure the text into clear paragraphs where natural breaks occur (e.g., \
between narrative sections, witness statements, evidence descriptions).
6. Use standard law-enforcement report conventions:
   - Times in 24-hour format when stated (e.g., 2345 hours).
   - Dates in MM/DD/YYYY format when stated.
   - Addresses, badge numbers, case numbers, and statute references formatted \
consistently.
7. Return ONLY the cleaned-up transcript. Do NOT include any commentary, \
preamble, or explanation.\
"""

# ---------------------------------------------------------------------------
# LFM2.5-Audio-1.5B configuration
# ---------------------------------------------------------------------------

_APP_DIR = os.path.dirname(os.path.abspath(__file__))

_lfm_path_env = os.environ.get("LFM_MODEL_PATH", "models/LFM2.5-Audio-1.5B")
LFM_MODEL_PATH: str = (
    _lfm_path_env
    if os.path.isabs(_lfm_path_env)
    else os.path.join(_APP_DIR, _lfm_path_env)
)

# Enforce offline mode when the local model directory exists so that
# liquid_audio / transformers never attempt to reach Hugging Face.
if os.path.isdir(LFM_MODEL_PATH):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Device selection: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    _device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    _device = "mps"
else:
    _device = "cpu"

_lfm_model = None
_lfm_processor = None
_lfm_lock = threading.Lock()

# ---------------------------------------------------------------------------
# faster-whisper fallback configuration
# ---------------------------------------------------------------------------

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
_whisper_path_env = os.environ.get("WHISPER_MODEL_PATH", "whisper-model")
WHISPER_MODEL_PATH: str = (
    _whisper_path_env
    if os.path.isabs(_whisper_path_env)
    else os.path.join(_APP_DIR, _whisper_path_env)
)
_whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
_whisper_compute_type = "float16" if _whisper_device == "cuda" else "int8"
_whisper_model: WhisperModel | None = None
_whisper_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _get_lfm():
    """Load LFM2.5-Audio-1.5B on first call (thread-safe).

    Loads from LFM_MODEL_PATH if the directory exists, otherwise falls back
    to downloading 'LiquidAI/LFM2.5-Audio-1.5B' from Hugging Face.
    """
    global _lfm_model, _lfm_processor
    if _lfm_model is None:
        with _lfm_lock:
            if _lfm_model is None:
                try:
                    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
                except ImportError as exc:
                    raise RuntimeError(
                        "The 'liquid-audio' package is not installed. "
                        "Run: pip install liquid-audio"
                    ) from exc
                model_source = (
                    LFM_MODEL_PATH if os.path.isdir(LFM_MODEL_PATH)
                    else "LiquidAI/LFM2.5-Audio-1.5B"
                )
                _lfm_processor = LFM2AudioProcessor.from_pretrained(model_source).eval()
                _lfm_model = LFM2AudioModel.from_pretrained(model_source).eval()
                if _device != "cpu":
                    _lfm_model = _lfm_model.to(_device)
    return _lfm_model, _lfm_processor


def _get_whisper_model() -> WhisperModel:
    """Return the faster-whisper model, loading it on first call (thread-safe).

    Loads from WHISPER_MODEL_PATH if the directory exists, otherwise falls
    back to downloading the WHISPER_MODEL_SIZE model from Hugging Face.
    """
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                model_source = (
                    WHISPER_MODEL_PATH if os.path.isdir(WHISPER_MODEL_PATH)
                    else WHISPER_MODEL_SIZE
                )
                try:
                    _whisper_model = WhisperModel(
                        model_source, device=_whisper_device, compute_type=_whisper_compute_type
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load faster-whisper model from '{model_source}'. "
                        f"Place a CTranslate2 Whisper model in '{WHISPER_MODEL_PATH}' "
                        f"or set WHISPER_MODEL_SIZE to a valid size "
                        f"(tiny, base, small, medium, large-v3). Detail: {exc}"
                    ) from exc
    return _whisper_model


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio_for_lfm(audio_path: str):
    """Load an audio file and return a mono 24 kHz waveform tensor.

    LFM2.5-Audio-1.5B performs best at 24 kHz mono input.
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 24000:
        waveform = torchaudio.transforms.Resample(sr, 24000)(waveform)
        sr = 24000
    return waveform, sr


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------

def _transcribe_lfm(audio_path: str) -> str:
    """Transcribe audio using LFM2.5-Audio-1.5B (returns the full transcript)."""
    from liquid_audio import ChatState, LFMModality

    model, processor = _get_lfm()
    waveform, sr = _load_audio_for_lfm(audio_path)

    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text("Perform ASR.")
    chat.end_turn()

    chat.new_turn("user")
    chat.add_audio(waveform, sr)
    chat.end_turn()

    transcript = ""
    for token, modality in model.generate_sequential(
        chat.model_input(_device),
        max_new_tokens=512,
    ):
        if modality == LFMModality.TEXT:
            transcript += processor.text.decode(token)

    return transcript.strip()


def _cleanup_lfm(raw_transcript: str) -> str:
    """Clean up a raw transcript using POLICE_REPORT_CLEANUP_PROMPT via LFM."""
    from liquid_audio import ChatState, LFMModality

    model, processor = _get_lfm()

    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(POLICE_REPORT_CLEANUP_PROMPT)
    chat.end_turn()

    chat.new_turn("user")
    chat.add_text(raw_transcript)
    chat.end_turn()

    cleaned = ""
    for token, modality in model.generate_sequential(
        chat.model_input(_device),
        max_new_tokens=2048,
    ):
        if modality == LFMModality.TEXT:
            cleaned += processor.text.decode(token)

    return cleaned.strip()


# ---------------------------------------------------------------------------
# Gradio handler
# ---------------------------------------------------------------------------

def asr_transcription(audio_input, apply_cleanup):
    """Transcribe audio and optionally apply the police-report cleanup prompt.

    Tries LFM2.5-Audio-1.5B first; falls back to faster-whisper if the LFM
    model is unavailable or raises an error.
    """
    if audio_input is None:
        yield "Please provide an audio input.", ""
        return

    try:
        yield "⏳ Transcribing…", ""

        # Primary: LFM2.5-Audio-1.5B
        try:
            raw = _transcribe_lfm(audio_input)
        except Exception as lfm_err:
            # Fallback: faster-whisper
            try:
                segments, _ = _get_whisper_model().transcribe(audio_input, beam_size=5)
                raw = " ".join(s.text for s in segments).strip()
            except Exception as w_err:
                yield (
                    f"LFM error: {lfm_err}\nWhisper fallback error: {w_err}",
                    "",
                )
                return

        if not raw:
            yield "No speech detected in the audio.", ""
            return

        if not apply_cleanup:
            yield raw, ""
            return

        yield raw, "⏳ Applying police report cleanup…"
        cleaned = _cleanup_lfm(raw)
        yield raw, cleaned

    except Exception as e:
        yield f"Error: {e}", ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def create_ui():
    """Create the Gradio interface."""
    with gr.Blocks(title="Liquid Audio — Offline ASR Dictation") as demo:
        gr.Markdown("# 🎙️ Liquid Audio — Offline ASR Dictation")
        gr.Markdown(
            "Offline speech-to-text dictation powered by **LFM2.5-Audio-1.5B**. "
            "Models are loaded from the local `models/` directory — no internet required."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Audio Input")
                audio_input = gr.Audio(
                    label="Upload Audio or Record",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                apply_cleanup = gr.Checkbox(
                    label="Apply Police Report Cleanup",
                    value=False,
                    info="Clean up the transcript using the police-report editing prompt.",
                )
                submit_btn = gr.Button("Transcribe", variant="primary", size="lg")

            with gr.Column():
                gr.Markdown("### Output")
                raw_output = gr.Textbox(
                    label="Raw Transcript",
                    interactive=False,
                    lines=8,
                )
                cleaned_output = gr.Textbox(
                    label="Cleaned Police Report",
                    interactive=False,
                    lines=12,
                )

        submit_btn.click(
            fn=asr_transcription,
            inputs=[audio_input, apply_cleanup],
            outputs=[raw_output, cleaned_output],
        )

        gr.Markdown("---")
        gr.Markdown(
            "**How to use:**\n"
            "1. Upload an audio file or record speech via the microphone\n"
            "2. Optionally check *Apply Police Report Cleanup* to format the transcript "
            "as a professional police report\n"
            "3. Click **Transcribe**\n\n"
            f"**LFM model path:** `{LFM_MODEL_PATH}` "
            "— see `LOCAL_MODEL_SETUP.md` for download instructions.\n\n"
            "Set `LFM_MODEL_PATH` to override the default path.  "
            "Set `WHISPER_MODEL_PATH` to use a local faster-whisper fallback model."
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Liquid Audio — Offline ASR Dictation"
    )
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="Run locally without creating a public shareable link (default: share is enabled).",
    )
    args = parser.parse_args()

    print(f"LFM model path   : {LFM_MODEL_PATH}")
    print(f"LFM model exists : {os.path.isdir(LFM_MODEL_PATH)}")
    print(f"Device           : {_device}")

    demo = create_ui()
    demo.launch(share=not args.no_share)

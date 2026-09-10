"""
app.py – Gradio UI for Perception-LM-1B (PLM) video understanding.

Runs facebook/Perception-LM-1B locally on the Hugging Face Space.
This mirrors tests/plm2.py exactly, but accepts user-uploaded videos.
"""
from __future__ import annotations

import os
import tempfile
import torch
from pathlib import Path

# Monkey-patch gradio_client JSON-schema bug (additionalProperties=True is bool)
import gradio_client.utils as _gc_utils

_orig_get_type = _gc_utils.get_type

def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _orig_get_type(schema)

_gc_utils.get_type = _patched_get_type

_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type

def _patched_json_schema_to_python_type(schema, defs):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)

_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type

import gradio as gr
from transformers import AutoProcessor, AutoModelForImageTextToText

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_PATH = os.environ.get("PLM_MODEL", "facebook/Perception-LM-1B")
NUM_FRAMES = int(os.environ.get("PLM_NUM_FRAMES", "32"))
MAX_NEW_TOKENS = int(os.environ.get("PLM_MAX_NEW_TOKENS", "256"))

# ------------------------------------------------------------------
# Lazy singleton for model/processor
# ------------------------------------------------------------------
_processor = None
_model = None


def get_processor():
    global _processor
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return _processor


def get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH).to(device)
    return _model


# ------------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------------
def analyze_video(video_path: str, text: str) -> dict:
    """Run PLM on a video + text prompt."""
    if video_path is None:
        return {"error": "Please upload a video file."}

    # Copy uploaded file to a stable path (Gradio gives a temp path)
    suffix = Path(video_path).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        import shutil
        shutil.copy(video_path, tmp.name)
        video_file = tmp.name

    try:
        processor = get_processor()
        model = get_model()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": video_file},
                    {"type": "text", "text": text},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            [conversation],
            num_frames=NUM_FRAMES,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            video_load_backend="decord",
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        input_length = inputs["input_ids"].shape[1]
        generate_ids_without_inputs = generate_ids[:, input_length:]

        outputs = processor.batch_decode(
            generate_ids_without_inputs, skip_special_tokens=True
        )

        return {
            "status": "ok",
            "model": MODEL_PATH,
            "device": str(model.device),
            "num_frames": NUM_FRAMES,
            "prompt": text,
            "generated_text": outputs[0] if outputs else "",
        }

    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)}"}
    finally:
        try:
            os.unlink(video_file)
        except Exception:
            pass


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------
with gr.Blocks(title="PLM Video Analysis") as demo:
    gr.Markdown("# Perception-LM-1B Video Understanding")
    gr.Markdown(
        "Runs `facebook/Perception-LM-1B` locally on the Space. "
        "The model is downloaded on first use."
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            text_input = gr.Textbox(
                label="Prompt",
                value="Can you describe the video in detail?",
                lines=2,
            )
            analyze_btn = gr.Button("Analyze", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Generated Text", lines=15)

    def _analyze(video, text):
        result = analyze_video(video, text)
        if "error" in result:
            return f"ERROR: {result['error']}"
        return result.get("generated_text", "No output")

    analyze_btn.click(
        fn=_analyze,
        inputs=[video_input, text_input],
        outputs=[output_text],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

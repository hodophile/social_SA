"""
app.py – Gradio UI for side-by-side comparison:
    * Initial Version  – Melisa POC (melisa_poc/, late fusion)
    * Timestamp-LLM    – per-frame SigLIP + timestamps → OpenRouter GPT-4o-mini

Both outputs are shown in the JSON format with execution_time_seconds,
understanding, risk, action, emotion_analysis, plus frame_timestamps for
the timestamp-LLM version.
"""
from __future__ import annotations

import os

# CPU-only Space fix: neutralise spaces.GPU before Melisa loads it
import spaces
spaces.GPU = lambda **kwargs: lambda f: f  # no-op decorator

from pathlib import Path

import gradio as gr

# ----------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------
from melisa_bridge import analyze_with_melisa              # Melisa POC
from llm_fusion_pipeline import TimestampLLMFusionPipeline  # new

KIMI_API_KEY = os.environ.get("KIMI_API_KEY")
KIMI_MODEL = os.environ.get("KIMI_MODEL", "kimi-k2-5")


# ----------------------------------------------------------------------
# Lazy singleton
# ----------------------------------------------------------------------
_llm_pipeline: TimestampLLMFusionPipeline | None = None


def get_llm_pipeline() -> TimestampLLMFusionPipeline:
    global _llm_pipeline
    if _llm_pipeline is None:
        _llm_pipeline = TimestampLLMFusionPipeline(
            kimi_api_key=KIMI_API_KEY,
            kimi_model=KIMI_MODEL,
        )
    return _llm_pipeline


# ----------------------------------------------------------------------
# Main entry: run both pipelines and compare
# ----------------------------------------------------------------------
def analyze(text: str, image: str, video: str, sample_fps: float):
    if not text and not image and not video:
        err = {"error": "Provide at least one of: text, image, video."}
        return err, err, {}

    # Melisa routes a single input (text XOR media) -> media wins.
    media_path = None
    if video:
        media_path = str(video)
    elif image:
        media_path = str(image)

    # ---------------- Initial Version (Melisa POC) ----------------
    melisa_res = analyze_with_melisa(
        text=None if media_path else (text or None),
        media_path=media_path,
    )

    # ---------------- Timestamp-LLM Version -----------------------
    llm_res = get_llm_pipeline().analyze(
        text=text or None,
        image_path=Path(image) if image else None,
        video_path=Path(video) if video else None,
        sample_fps=sample_fps,
    )

    # ---------------- Comparison summary -------------------------
    llm_emo = llm_res.get("emotion_analysis", {})

    comparison = {
        "initial_version": {
            "matched_emotion": melisa_res.get("matched_emotion"),
            "initial_label": melisa_res.get("initial_label"),
            "valence": melisa_res.get("valence"),
            "confidence": melisa_res.get("initial_confidence"),
        },
        "timestamp_llm_version": {
            "primary_emotion": llm_emo.get("primary_emotion"),
            "primary_emotion_score": llm_emo.get(
                "primary_emotion_score"
            ),
            "valence": llm_emo.get("valence"),
            "confidence": llm_emo.get("confidence"),
        },
    }

    v1_emo = comparison["initial_version"]["matched_emotion"]
    v2_emo = comparison["timestamp_llm_version"]["primary_emotion"]

    if v1_emo and v2_emo:
        comparison["emotions_match"] = v1_emo == v2_emo

        v1_val = comparison["initial_version"]["valence"]
        v2_val = comparison["timestamp_llm_version"]["valence"]
        if v1_val is not None and v2_val is not None:
            comparison["valence_gap"] = round(abs(v1_val - v2_val), 3)

    return melisa_res, llm_res, comparison


# ----------------------------------------------------------------------
# UI text
# ----------------------------------------------------------------------
DESCRIPTION = (
    "Side-by-side comparison of two sentiment/emotion pipelines.\n\n"

    "**Initial Version — Melisa POC** (per-modality models + late fusion):\n"
    "- Frames scored independently (SigLIP zero-shot), then averaged — "
    "**no temporal model**\n"
    "- Audio: Whisper ASR -> RoBERTa sentiment on the transcript\n"
    "- Caption: RoBERTa sentiment\n"
    "- Confidence-weighted late fusion -> positive / neutral / negative, "
    "mapped onto the emotion labels (positive->joy, neutral->trust, "
    "negative->sadness)\n\n"

    "**Timestamp-LLM Version** (per-frame + LLM synthesis):\n"
    "- Video → extract frames at sample_fps with timestamps\n"
    "- Each frame → SigLIP zero-shot 8-emotion classification\n"
    "- Audio → Faster-Whisper → transcript\n"
    "- **ALL context sent to Kimi (Moonshot AI)**: frame emotions + "
    "timestamps + audio transcript + caption\n"
    "- LLM synthesizes temporal dynamics and cross-modal agreement\n"
    "- Returns full 8-emotion distribution with rationale\n"
)

with gr.Blocks(
    title="Initial Version (Melisa) vs Timestamp-LLM-Fusion",
    theme=gr.themes.Soft(primary_hue="indigo"),
) as demo:

    gr.Markdown("# Social Media Post Analysis — Timestamp-LLM vs Melisa")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        text_in = gr.Textbox(label="Post text / caption", lines=3)
        image_in = gr.Image(label="Image (optional)", type="filepath")
        video_in = gr.Video(label="Video (optional)")
        fps_in = gr.Slider(
            0.2, 5.0, value=1.0, step=0.2,
            label="Video frame sampling (fps)",
        )

    analyze_btn = gr.Button("Analyze with both", variant="primary")

    gr.Markdown("## Comparison")
    comparison_out = gr.JSON(label="Side-by-side summary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Initial Version (Melisa POC, late fusion)")
            melisa_out = gr.JSON(
                label="Initial Version sentiment (labels matched to emotions)"
            )
        with gr.Column():
            gr.Markdown("### Timestamp-LLM Version (SigLIP + OpenRouter)")
            llm_out = gr.JSON(label="Timestamp-LLM result")

    analyze_btn.click(
        fn=analyze,
        inputs=[text_in, image_in, video_in, fps_in],
        outputs=[melisa_out, llm_out, comparison_out],
    )

    gr.Examples(
        examples=[
            ["I am so excited about this new project!", None, None, 1.0],
        ],
        inputs=[text_in, image_in, video_in, fps_in],
        outputs=[melisa_out, llm_out, comparison_out],
        fn=analyze,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)

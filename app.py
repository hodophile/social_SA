"""
app.py – Gradio UI for side-by-side comparison:
    * Initial Version  – original Melisa POC (melisa_poc/, late fusion)
    * Current Version  – new Temporal-aware 8-emotion pipeline
      (melisa_temporal_emotion/)

Both outputs are shown in the JSON format used by the Qwen2.5-VLM-3B
setup (execution_time_seconds / understanding / risk / action /
emotion_analysis).
"""
from __future__ import annotations

import os
# Use CPU for Melisa POC to avoid ZeroGPU CUDA issues
os.environ.setdefault("DEVICE", "cpu")

from pathlib import Path

import gradio as gr

# ----------------------------------------------------------------------
# Pipelines
# ----------------------------------------------------------------------
from melisa_bridge import analyze_with_melisa              # Melisa POC
from melisa_temporal_emotion.pipeline import (             # new pipeline
    TemporalEmotionPipeline,
)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")


# ----------------------------------------------------------------------
# Lazy singleton for the temporal pipeline
# ----------------------------------------------------------------------
_temporal_pipeline: TemporalEmotionPipeline | None = None


def get_temporal_pipeline() -> TemporalEmotionPipeline:
    global _temporal_pipeline
    if _temporal_pipeline is None:
        _temporal_pipeline = TemporalEmotionPipeline(
            openrouter_api_key=OPENROUTER_API_KEY,
            openrouter_model=OPENROUTER_MODEL,
        )
    return _temporal_pipeline


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

    # ---------------- Current Version (Temporal Emotion) ----------
    temporal_res = get_temporal_pipeline().analyze(
        text=text or None,
        image_path=Path(image) if image else None,
        video_path=Path(video) if video else None,
        sample_fps=sample_fps,
    )

    # ---------------- Comparison summary -------------------------
    temporal_emo = temporal_res.get("emotion_analysis", {})

    comparison = {
        "initial_version": {
            "matched_emotion": melisa_res.get("matched_emotion"),
            "initial_label": melisa_res.get("initial_label"),
            "valence": melisa_res.get("valence"),
            "confidence": melisa_res.get("initial_confidence"),
        },
        "current_version": {
            "primary_emotion": temporal_emo.get("primary_emotion"),
            "primary_emotion_score": temporal_emo.get(
                "primary_emotion_score"
            ),
            "valence": temporal_emo.get("valence"),
            "confidence": temporal_emo.get("confidence"),
        },
    }

    v1_emo = comparison["initial_version"]["matched_emotion"]
    v2_emo = comparison["current_version"]["primary_emotion"]

    if v1_emo and v2_emo:
        comparison["emotions_match"] = v1_emo == v2_emo

        v1_val = comparison["initial_version"]["valence"]
        v2_val = comparison["current_version"]["valence"]
        if v1_val is not None and v2_val is not None:
            comparison["valence_gap"] = round(abs(v1_val - v2_val), 3)

    return melisa_res, temporal_res, comparison


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

    "**Current Version — Temporal Emotion** (new):\n"
    "- Per-frame 8-emotion CNN (ResNet-18 / FER2013 -> Plutchik's 8)\n"
    "- **Temporal aggregation**: exponential-decay weighting over the frame "
    "sequence, so order and transitions matter\n"
    "- Audio transcript -> OpenRouter 8-emotion analysis (RoBERTa fallback)\n"
    "- Same confidence-weighted late fusion, but fusing full 8-emotion "
    "distributions instead of bare positive/negative scores\n"
)

with gr.Blocks(
    title="Initial Version (Melisa) vs Current Version (Temporal Emotion)",
    theme=gr.themes.Soft(primary_hue="indigo"),
) as demo:

    gr.Markdown("# Social Media Post Analysis — Comparative Deployment")
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
            gr.Markdown("### Current Version (Temporal Emotion)")
            temporal_out = gr.JSON(label="Current Version result")

    analyze_btn.click(
        fn=analyze,
        inputs=[text_in, image_in, video_in, fps_in],
        outputs=[melisa_out, temporal_out, comparison_out],
    )

    gr.Examples(
        examples=[
            ["I am so excited about this new project!", None, None, 1.0],
        ],
        inputs=[text_in, image_in, video_in, fps_in],
        outputs=[melisa_out, temporal_out, comparison_out],
        fn=analyze,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
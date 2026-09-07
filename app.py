"""
app.py – Gradio UI for side‑by‑side comparison:
    * Original Melisa POC (melisa_poc/)
    * New Temporal‑aware 8‑emotion pipeline (melisa_temporal_emotion/)
Both return JSON in the format expected by the Qwen+OpenRouter setup.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import gradio as gr

# ----------------------------------------------------------------------
# Import the two pipelines
# ----------------------------------------------------------------------
import sys
sys.path.append(str(Path(__file__).parent / "melisa_poc"))
from src.pipeline import MyUniSentimentPipeline   # original Melisa

from melisa_temporal_emotion.pipeline import TemporalEmotionPipeline   # new

# ----------------------------------------------------------------------
# Configuration – defaults (can be overridden via Space secrets / env vars)
# ----------------------------------------------------------------------
MOCK = os.environ.get("PLM_MOCK", "0") == "1"          # 0 = real model, 1 = mock (not used by Melisa pipeline)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")  # kept for compatibility but not used

# ----------------------------------------------------------------------
# Lazy‑loaded pipeline singletons
# ----------------------------------------------------------------------
_melisa_pipeline: Optional[MyUniSentimentPipeline] = None
_temporal_pipeline: Optional[TemporalEmotionPipeline] = None


def get_melisa_pipeline() -> MyUniSentimentPipeline:
    global _melisa_pipeline
    if _melisa_pipeline is None:
        _melisa_pipeline = MyUniSentimentPipeline(
            # No model_path or mock arguments; MyUniSentimentPipeline uses defaults.
        )
    return _melisa_pipeline


def get_temporal_pipeline() -> TemporalEmotionPipeline:
    global _temporal_pipeline
    if _temporal_pipeline is None:
        _temporal_pipeline = TemporalEmotionPipeline(
            openrouter_api_key=OPENROUTER_API_KEY,
            openrouter_model=OPENROUTER_MODEL,
        )
    return _temporal_pipeline


# ----------------------------------------------------------------------
# Core analyse function – runs both pipelines and returns a tuple
# ----------------------------------------------------------------------
def analyze(text: str, image: str, video: str, sample_fps: float):
    if not text and not image and not video:
        err = {"error": "Provide at least one of: text, image, video."}
        return err, err, err, {}   # four outputs: melisa, temporal, understanding, comparison

    post = {
        "text": text or None,
        "image_path": Path(image) if image else None,
        "video_path": Path(video) if video else None,
    }

    # ------------------------------------------------------------------
    # Run Melisa POC
    # ------------------------------------------------------------------
    melisa_res = get_melisa_pipeline().analyze(
        text=post["text"],
        image_path=post["image_path"],
        video_path=post["video_path"],
        sample_fps=sample_fps,
    )

    # ------------------------------------------------------------------
    # Run Temporal Emotion pipeline
    # ------------------------------------------------------------------
    temporal_res = get_temporal_pipeline().analyze(
        text=post["text"],
        image_path=post["image_path"],
        video_path=post["video_path"],
        sample_fps=sample_fps,
    )

    # ------------------------------------------------------------------
    # Build comparison summary
    # ------------------------------------------------------------------
    comparison = {
        "melisa": {
            "primary_emotion": melisa_res.get("emotion_analysis", {}).get("primary_emotion"),
            "primary_emotion_score": melisa_res.get("emotion_analysis", {}).get("primary_emotion_score"),
            "valence": melisa_res.get("emotion_analysis", {}).get("valence"),
            "confidence": melisa_res.get("emotion_analysis", {}).get("confidence"),
        },
        "temporal": {
            "primary_emotion": temporal_res.get("emotion_analysis", {}).get("primary_emotion"),
            "primary_emotion_score": temporal_res.get("emotion_analysis", {}).get("primary_emotion_score"),
            "valence": temporal_res.get("emotion_analysis", {}).get("valence"),
            "confidence": temporal_res.get("emotion_analysis", {}).get("confidence"),
        },
    }

    # Determine if the primary emotions match
    melisa_primary = comparison["melisa"]["primary_emotion"]
    temporal_primary = comparison["temporal"]["primary_emotion"]
    if melisa_primary and temporal_primary:
        comparison["emotions_match"] = melisa_primary == temporal_primary

        melisa_val = comparison["melisa"]["valence"]
        temporal_val = comparison["temporal"]["valence"]
        if melisa_val is not None and temporal_val is not None:
            comparison["valence_gap"] = round(abs(melisa_val - temporal_val), 3)

    # ------------------------------------------------------------------
    # Return four values matching the Gradio outputs:
    #   melisa JSON, temporal JSON, understanding (we reuse melisa's understanding as a proxy),
    #   comparison summary
    # ------------------------------------------------------------------
    # For the "understanding" column we simply show the Melisa understanding –
    # it already contains content, tone, intent, visual/audio description, etc.
    # You could also show the temporal pipeline's understanding if you prefer.
    return melisa_res, temporal_res, melisa_res.get("understanding", {}), comparison


# ----------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------
DESCRIPTION = (
    "Side‑by‑side comparison of two sentiment pipelines that both return "
    "the Qwen + OpenRouter JSON format.\\n\\n"
    "**Left – Melisa POC** (the original vendored pipeline):\\n"
    " • Frame‑level SigLIP visual sentiment → confidence‑weighted average (no temporal model)\\n"
    " • Whisper → RoBERTa text sentiment for audio transcript\\n"
    " • Caption → RoBERTa sentiment\\n"
    " • Late fusion of the three modalities → 3‑class label (positive/neutral/negative) mapped to our emotion set.\\n\\n"
    "**Right – Temporal Emotion** (new pipeline):\\n"
    " • Per‑frame 8‑emotion CNN (ResNet‑18 → FER2013) → exponential‑decay temporal weighting\\n"
    " • Audio transcript → OpenRouter emotion analysis (or RoBERTa fallback)\\n"
    " • Caption → heuristic mapping of RoBERTa sentiment to 8 emotions\\n"
    " • Same confidence‑weighted late fusion (`fuse_modalities`) → 8‑emotion distribution, valence, confidence.\\n"
)

with gr.Blocks(
    title="Melisa POC vs Temporal Emotion (side‑by‑side)",
    theme=gr.themes.Soft(primary_hue="indigo"),
) as demo:
    gr.Markdown("# Social Media Post Analysis – Side‑by‑Side Comparison")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        text_in = gr.Textbox(label="Post text / caption", lines=3)
        image_in = gr.Image(label="Image (optional)", type="filepath")
        video_in = gr.Video(label="Video (optional)")
        fps_in = gr.Slider(
            0.2, 5.0, value=1.0, step=0.2,
            label="Video frame sampling (fps)",
        )

    analyze_btn = gr.Button("Analyze with both pipelines", variant="primary")

    gr.Markdown("## Comparison")
    comparison_out = gr.JSON(label="Side‑by‑side summary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Melisa POC (original)")
            melisa_out = gr.JSON(label="Melisa result")
        with gr.Column():
            gr.Markdown("### Temporal Emotion (new)")
            temporal_out = gr.JSON(label="Temporal Emotion result")

    analyze_btn.click(
        fn=analyze,
        inputs=[text_in, image_in, video_in, fps_in],
        outputs=[melisa_out, temporal_out, comparison_out],
    )

    gr.Examples(
        examples=[
            ["I am so excited about this new project!", None, None, 1.0],
            ["Check out this clip!", None, "examples/assets/sample_video.mp4", 1.0],
            [None, "examples/assets/sample_image.png", None, 1.0],
        ],
        inputs=[text_in, image_in, video_in, fps_in],
        outputs=[melisa_out, temporal_out, comparison_out],
        fn=analyze,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)

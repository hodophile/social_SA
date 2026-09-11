"""
app.py – Gradio UI for VideoPrism video understanding.

Uses google/videoprism-base-f16r288 for video feature extraction
and zero-shot classification with text prompts for:
- Video content explanation
- Emotion detection
- Video classification
"""
from __future__ import annotations

import os
import tempfile
import torch
import torch.nn.functional as F
from pathlib import Path

import gradio as gr
from transformers import AutoModel, AutoVideoProcessor
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_PATH = os.environ.get("VIDEOPRISM_MODEL", "google/videoprism-base-f16r288")
TEXT_MODEL_PATH = os.environ.get("TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
NUM_FRAMES = int(os.environ.get("VIDEOPRISM_NUM_FRAMES", "16"))

# Predefined labels for zero-shot classification
VIDEO_CATEGORIES = [
    "sports and fitness",
    "cooking and food",
    "music and dance",
    "travel and nature",
    "technology and gadgets",
    "gaming",
    "education and tutorial",
    "news and documentary",
    "comedy and entertainment",
    "art and creativity",
]

EMOTION_LABELS = [
    "joy and happiness",
    "sadness and melancholy",
    "anger and frustration",
    "fear and anxiety",
    "surprise and shock",
    "disgust and revulsion",
    "trust and comfort",
    "anticipation and excitement",
    "neutral and calm",
]

# ------------------------------------------------------------------
# Lazy singleton for models
# ------------------------------------------------------------------
_video_processor = None
_video_model = None
_text_model = None


def get_video_processor():
    global _video_processor
    if _video_processor is None:
        _video_processor = AutoVideoProcessor.from_pretrained(MODEL_PATH)
    return _video_processor


def get_video_model():
    global _video_model
    if _video_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _video_model = AutoModel.from_pretrained(MODEL_PATH).to(device)
        _video_model.eval()
    return _video_model


def get_text_model():
    global _text_model
    if _text_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _text_model = SentenceTransformer(TEXT_MODEL_PATH, device=device)
    return _text_model


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------
def extract_video_features(video_path: str):
    """Extract pooled video embedding from VideoPrism."""
    processor = get_video_processor()
    model = get_video_model()

    processed = processor(
        videos=[video_path],
        return_metadata=True,
        do_sample_frames=True,
    )
    pixel_values = processed["pixel_values_videos"].to(model.device)

    with torch.no_grad():
        outputs = model(pixel_values)

    # Use mean pooling of last hidden state
    last_hidden = outputs.last_hidden_state
    pooled = last_hidden.mean(dim=1)
    return F.normalize(pooled, p=2, dim=1)


def extract_text_features(texts: list[str]):
    """Extract text embeddings from sentence-transformers."""
    model = get_text_model()
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    return F.normalize(embeddings, p=2, dim=1)


# ------------------------------------------------------------------
# Zero-shot classification
# ------------------------------------------------------------------
def zero_shot_classify(video_embedding, candidate_labels: list[str]):
    """Return best matching label and similarity scores."""
    text_embeddings = extract_text_features(candidate_labels)
    similarities = torch.mm(video_embedding, text_embeddings.T).squeeze(0)
    probs = F.softmax(similarities / 0.1, dim=0)

    best_idx = int(similarities.argmax())
    scores = {
        label: round(float(prob), 4)
        for label, prob in zip(candidate_labels, probs.cpu().numpy())
    }
    return candidate_labels[best_idx], scores


# ------------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------------
def analyze_video(video_path: str, text_prompt: str) -> dict:
    """Run VideoPrism analysis on a video."""
    if video_path is None:
        return {"error": "Please upload a video file."}

    # Copy uploaded file to a stable path
    suffix = Path(video_path).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        import shutil
        shutil.copy(video_path, tmp.name)
        video_file = tmp.name

    try:
        # Extract video features
        video_emb = extract_video_features(video_file)

        # 1. Video Classification
        category, cat_scores = zero_shot_classify(video_emb, VIDEO_CATEGORIES)

        # 2. Emotion Detection
        emotion, emo_scores = zero_shot_classify(video_emb, EMOTION_LABELS)

        # 3. Custom prompt classification (if provided)
        custom_result = None
        custom_scores = None
        if text_prompt and text_prompt.strip():
            # Treat prompt as a question and compare with yes/no
            custom_labels = [f"yes, {text_prompt}", f"no, {text_prompt}"]
            custom_result, custom_scores = zero_shot_classify(video_emb, custom_labels)

        # Build explanation
        top_categories = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_emotions = sorted(emo_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        explanation = (
            f"This video appears to be about **{category}**.\n\n"
            f"**Emotional Tone:** {emotion}\n\n"
            f"**Top Categories:**\n"
        )
        for cat, score in top_categories:
            explanation += f"  - {cat}: {score:.1%}\n"

        explanation += "\n**Top Emotions:**\n"
        for emo, score in top_emotions:
            explanation += f"  - {emo}: {score:.1%}\n"

        if custom_result:
            explanation += f"\n**Custom Prompt ('{text_prompt}'):** {custom_result}\n"

        return {
            "status": "ok",
            "model": MODEL_PATH,
            "device": str(get_video_model().device),
            "content_category": category,
            "category_confidence": round(cat_scores[category], 4),
            "emotion": emotion,
            "emotion_confidence": round(emo_scores[emotion], 4),
            "top_categories": top_categories,
            "top_emotions": top_emotions,
            "custom_prompt_result": custom_result,
            "custom_prompt_scores": custom_scores,
            "explanation": explanation,
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
with gr.Blocks(title="VideoPrism Video Analysis") as demo:
    gr.Markdown("# VideoPrism Video Understanding")
    gr.Markdown(
        "Uses `google/videoprism-base-f16r288` for video feature extraction "
        "and zero-shot classification for content, emotion, and custom prompts."
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            text_input = gr.Textbox(
                label="Custom Prompt (Optional)",
                placeholder="e.g., Is this video suitable for children?",
                lines=2,
            )
            analyze_btn = gr.Button("Analyze", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Analysis Result", lines=20)

    def _analyze(video, text):
        result = analyze_video(video, text)
        if "error" in result:
            return f"ERROR: {result['error']}"
        return result.get("explanation", "No output")

    analyze_btn.click(
        fn=_analyze,
        inputs=[video_input, text_input],
        outputs=[output_text],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

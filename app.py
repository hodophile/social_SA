"""
app.py – Gradio UI for VideoPrism video-text matching.
Follows the official Google DeepMind VideoPrism Colab demo:
https://colab.research.google.com/github/google-deepmind/videoprism/blob/main/videoprism/colabs/videoprism_video_text_demo.ipynb

Uses JAX/Flax model from the videoprism repository for joint video-text encoding.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import jax
import jax.numpy as jnp
import mediapy
import numpy as np
from PIL import Image

# Import VideoPrism from the installed package
try:
    from videoprism import models as vp
except ImportError:
    # Fallback: if package not installed, try to import from local clone
    import sys
    sys.path.append("./videoprism_repo")
    from videoprism import models as vp

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME = os.environ.get("VIDEOPRISM_MODEL", "videoprism_lvt_public_v1_base")
NUM_FRAMES = int(os.environ.get("VIDEOPRISM_NUM_FRAMES", "16"))
FRAME_SIZE = int(os.environ.get("VIDEOPRISM_FRAME_SIZE", "288"))
TEMPERATURE = float(os.environ.get("VIDEOPRISM_TEMPERATURE", "0.01"))
TOP_K = int(os.environ.get("VIDEOPRISM_TOP_K", "5"))

# Predefined labels for zero-shot classification (from original app)
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
]

# Combine all labels for zero-shot classification
ZERO_SHOT_LABELS = VIDEO_CATEGORIES + EMOTION_LABELS

# ------------------------------------------------------------------
# Video preprocessing (from Colab)
# ------------------------------------------------------------------
def read_and_preprocess_video(
    filename: str, target_num_frames: int, target_frame_size: tuple[int, int]
):
    """Reads and preprocesses a video exactly like the Colab demo."""
    frames = mediapy.read_video(filename)
    
    # Sample to target number of frames.
    frame_indices = np.linspace(
        0, len(frames), num=target_num_frames, endpoint=False, dtype=np.int32
    )
    frames = np.array([frames[i] for i in frame_indices])
    
    # Resize to target size.
    original_height, original_width = frames.shape[-3:-1]
    target_height, target_width = target_frame_size
    assert (
        original_height * target_width == original_width * target_height
    ), "Currently does not support aspect ratio mismatch."
    frames = mediapy.resize_video(frames, shape=target_frame_size)
    
    # Normalize pixel values to [0.0, 1.0].
    frames = mediapy.to_float01(frames)
    
    return frames

# ------------------------------------------------------------------
# Text preprocessing (from Colab)
# ------------------------------------------------------------------
def prepare_text_queries(text_queries, prompt_template="a video of {}."):
    """Prepare text queries with prompt template like the Colab."""
    return [prompt_template.format(t) for t in text_queries]

# ------------------------------------------------------------------
# Lazy singleton for model and tokenizer
# ------------------------------------------------------------------
_flax_model = None
_loaded_state = None
_text_tokenizer = None
_forward_fn = None

def get_videoprism_model():
    """Load and initialize the VideoPrism model (JAX/Flax)."""
    global _flax_model, _loaded_state, _text_tokenizer, _forward_fn
    
    if _flax_model is None:
        # Load model
        _flax_model = vp.get_model(MODEL_NAME, fprop_dtype=None)  # FP32 by default
        
        # Load pretrained weights
        _loaded_state = vp.load_pretrained_weights(MODEL_NAME)
        
        # Load text tokenizer
        _text_tokenizer = vp.load_text_tokenizer('c4_en')
        
        # Define and JIT-compile forward function
        @jax.jit
        def forward_fn(inputs, text_token_ids, text_paddings, train=False):
            return _flax_model.apply(
                _loaded_state,
                inputs,
                text_token_ids,
                text_paddings,
                train=train,
            )
        
        _forward_fn = forward_fn
    
    return _flax_model, _loaded_state, _text_tokenizer, _forward_fn

# ------------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------------
def analyze_video(video_path: str, text_prompt: str = "") -> dict:
    """Run VideoPrism video-text matching."""
    if video_path is None:
        return {"error": "Please upload a video file."}

    # Copy uploaded file to a stable path
    suffix = Path(video_path).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        import shutil
        shutil.copy(video_path, tmp.name)
        video_file = tmp.name

    try:
        # Load model
        flax_model, loaded_state, text_tokenizer, forward_fn = get_videoprism_model()
        
        # Preprocess video (like Colab)
        frames = read_and_preprocess_video(
            video_file, 
            target_num_frames=NUM_FRAMES, 
            target_frame_size=(FRAME_SIZE, FRAME_SIZE)
        )
        frames = jnp.asarray(frames[None, ...])  # Add batch dimension
        
        # Prepare text queries
        if text_prompt.strip():
            # Use user-provided prompt as a single query
            text_queries = [text_prompt]
        else:
            # Use predefined zero-shot labels
            text_queries = ZERO_SHOT_LABELS
        
        # Apply prompt template (like Colab: 'a video of {}.')
        text_queries = prepare_text_queries(text_queries, "a video of {}.")
        
        # Tokenize text
        text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, text_queries)
        
        # Compute embeddings
        video_embeddings, text_embeddings, _ = forward_fn(
            frames, text_ids, text_paddings, train=False
        )
        
        # Compute similarity matrix (like Colab)
        similarity_matrix = np.dot(
            np.array(video_embeddings), 
            np.array(text_embeddings).T
        )
        
        # Apply temperature
        similarity_matrix /= TEMPERATURE
        
        # Apply softmax over texts (to get probabilities)
        similarity_matrix = np.exp(similarity_matrix)
        similarity_matrix = similarity_matrix / np.sum(similarity_matrix, axis=1, keepdims=True)
        
        # Get results for the first (and only) video
        similarity_vector = similarity_matrix[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarity_vector)[::-1][:TOP_K]
        
        # Format results
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            label = text_queries[idx].replace("a video of ", "").rstrip(".")
            score = float(similarity_vector[idx])
            results.append({
                "rank": rank,
                "label": label,
                "similarity": score,
                "percentage": f"{score*100:.1f}%"
            })
        
        # Determine if we used custom prompt or zero-shot
        used_custom_prompt = bool(text_prompt.strip())
        
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "frames_used": NUM_FRAMES,
            "frame_size": FRAME_SIZE,
            "temperature": TEMPERATURE,
            "used_custom_prompt": used_custom_prompt,
            "results": results,
            "top_prediction": results[0] if results else None,
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
with gr.Blocks(title="VideoPrism Video-Text Matching") as demo:
    gr.Markdown("# VideoPrism Video-Text Understanding")
    gr.Markdown(
        "Uses the official [VideoPrism](https://github.com/google-deepmind/videoprism) "
        "video foundation model for joint video-text encoding. "
        "Follows the [Colab demo](https://colab.research.google.com/github/google-deepmind/videoprism/blob/main/videoprism/colabs/videoprism_video_text_demo.ipynb)."
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            text_input = gr.Textbox(
                label="Optional: Custom text prompt (otherwise uses predefined categories)",
                placeholder="e.g., 'a person playing guitar' or leave empty for zero-shot classification",
                lines=2,
            )
            analyze_btn = gr.Button("Analyze Video", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Results", lines=20)
            # Alternative: use JSON for structured output
            # output_json = JSON(label="Results")

    def _analyze(video, text):
        result = analyze_video(video, text)
        if "error" in result:
            return f"ERROR: {result['error']}"
        
        if result["status"] != "ok":
            return f"Unexpected error: {result}"
        
        # Format output for display
        output_lines = []
        output_lines.append(f"Model: {result['model']}")
        output_lines.append(f"Frames: {result['frames_used']} @ {result['frame_size']}x{result['frame_size']}")
        output_lines.append(f"Temperature: {result['temperature']}")
        output_lines.append("")
        
        if result["used_custom_prompt"]:
            output_lines.append("Using custom text prompt:")
            output_lines.append(f"  > {text}")
            output_lines.append("")
        else:
            output_lines.append("Using zero-shot classification with predefined categories")
            output_lines.append("")
        
        output_lines.append("Top matches:")
        output_lines.append("-" * 40)
        
        for res in result["results"]:
            output_lines.append(
                f"{res['rank']}. {res['label']:<30} {res['similarity']:.4f} ({res['percentage']})"
            )
        
        output_lines.append("")
        output_lines.append(f"Best match: {result['top_prediction']['label']} "
                          f"({result['top_prediction']['similarity']:.3f})")
        
        return "\n".join(output_lines)

    analyze_btn.click(
        fn=_analyze,
        inputs=[video_input, text_input],
        outputs=[output_text],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
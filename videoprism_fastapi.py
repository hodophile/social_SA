"""
videoprism_fastapi.py – Simple FastAPI service for VideoPrism video-text matching.
Follows the official Google DeepMind VideoPrism Colab demo.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

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
]

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
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="VideoPrism Video-Text Matching API")

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "VideoPrism Video-Text Matching",
        "model": MODEL_NAME,
        "description": "Joint video-text embedding for zero-shot video classification/retrieval",
        "endpoints": {
            "GET /": "This information",
            "POST /analyze": "Analyze video with optional text prompt (returns JSON)"
        },
        "example_usage": {
            "curl": "curl -X POST 'http://localhost:7860/analyze' -F 'video=@myvideo.mp4' -F 'text=a person playing guitar'",
            "python": "requests.post('http://localhost:7860/analyze', files={'video': open('myvideo.mp4', 'rb')}, data={'text': 'a person playing guitar'})"
        }
    }

@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    text: Optional[str] = Form(None)
):
    """
    Analyze a video with VideoPrism.
    
    Args:
        video: Video file (mp4, avi, mov, etc.)
        text: Optional text prompt. If empty or not provided, uses zero-shot classification
              with predefined categories (sports, cooking, emotions, etc.)
    
    Returns:
        JSON with analysis results including top-k matches and similarity scores
    """
    # Validate file type
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided")
    
    # Save uploaded video to temporary file
    suffix = Path(video.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
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
        if text and text.strip():
            # Use user-provided prompt as a single query
            text_queries = [text]
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
        used_custom_prompt = bool(text and text.strip())
        
        return JSONResponse(content={
            "status": "ok",
            "model": MODEL_NAME,
            "frames_used": NUM_FRAMES,
            "frame_size": FRAME_SIZE,
            "temperature": TEMPERATURE,
            "used_custom_prompt": used_custom_prompt,
            "results": results,
            "top_prediction": results[0] if results else None,
        })

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {str(exc)}")
    finally:
        # Clean up temporary file
        try:
            os.unlink(video_file)
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
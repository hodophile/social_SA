"""
FastAPI service for Perception-LM-1B (PLM) video understanding.
Uses Hugging Face Inference Client instead of raw requests.

Endpoints:
  POST /analyze       — analyze video + text with PLM via HF Inference API
  POST /analyze-text  — text-only inference (always works)
  GET  /health        — health check

Requires HF_TOKEN environment variable.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from huggingface_hub import InferenceClient

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
PLM_MODEL = "facebook/Perception-LM-1B"
VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Initialize HF Inference Client (handles endpoints, retries, etc.)
client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="PLM Video Analysis (HF Inference Client)", version="1.1.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": PLM_MODEL,
        "hf_token_set": bool(HF_TOKEN),
        "client_initialized": client is not None,
    }


@app.post("/analyze-text")
async def analyze_text(
    text: str = Form(...),
    max_new_tokens: int = Form(256),
):
    """Text-only inference — always works via HF Inference API."""
    if not client:
        return JSONResponse(
            status_code=500,
            content={"error": "HF_TOKEN not set."},
        )

    try:
        result = client.chat_completion(
            messages=[{"role": "user", "content": text}],
            model=PLM_MODEL,
            max_tokens=max_new_tokens,
            temperature=0.7,
        )
        return {
            "status": "ok",
            "model": PLM_MODEL,
            "prompt": text,
            "generated_text": result.choices[0].message.content,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"{type(exc).__name__}: {str(exc)}"},
        )


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    text: str = Form("Describe this video in detail"),
    max_new_tokens: int = Form(256),
    sample_fps: float = Form(1.0),
    max_frames: int = Form(8),
):
    """
    Analyze video + text using frame extraction + VLM.

    Steps:
      1. Extract frames from video using ffmpeg
      2. Send frames to VLM (Qwen2.5-VL) via HF Inference Client
      3. Aggregate descriptions with LLM
    """
    if not client:
        return JSONResponse(
            status_code=500,
            content={"error": "HF_TOKEN not set."},
        )

    # Save uploaded video
    suffix = Path(video.filename or "video.mp4").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await video.read())
        video_path = tmp.name

    try:
        # Extract frames
        frames = _extract_frames(video_path, sample_fps, max_frames)
        if not frames:
            return JSONResponse(
                status_code=400,
                content={"error": "No frames could be extracted from video."},
            )

        # Analyze each frame with VLM
        frame_descriptions = []
        for i, frame_path in enumerate(frames):
            desc = _describe_frame(frame_path, text, i)
            frame_descriptions.append({
                "frame_index": i,
                "timestamp_seconds": round(i / sample_fps, 2) if sample_fps > 0 else i,
                "description": desc,
            })

        # Aggregate with LLM
        aggregated = _aggregate_descriptions(frame_descriptions, text)

        return {
            "status": "ok",
            "vlm_model": VLM_MODEL,
            "llm_model": LLM_MODEL,
            "prompt": text,
            "num_frames": len(frames),
            "frame_descriptions": frame_descriptions,
            "aggregated_description": aggregated,
        }

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"{type(exc).__name__}: {str(exc)}"},
        )
    finally:
        # Cleanup
        try:
            os.unlink(video_path)
            for f in frames:
                os.unlink(f)
        except Exception:
            pass


def _extract_frames(video_path: str, sample_fps: float, max_frames: int) -> list:
    """Extract frames from video using ffmpeg. Returns list of temp file paths."""
    # Get duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip()) if probe.returncode == 0 else 10.0

    num_frames = min(max_frames, int(duration * sample_fps))
    if num_frames < 1:
        num_frames = 1

    frames = []
    for i in range(num_frames):
        timestamp = (i / max(num_frames - 1, 1)) * duration
        frame_path = f"/tmp/plm_frame_{i:04d}.png"
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
             "-frames:v", "1", "-f", "image2", frame_path],
            capture_output=True,
        )
        if result.returncode == 0 and Path(frame_path).exists():
            frames.append(frame_path)

    return frames


def _describe_frame(frame_path: str, prompt: str, frame_idx: int) -> str:
    """Send a single frame to VLM for description."""
    try:
        # Use conversational task with image + text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": f"file://{frame_path}"},
                    {"type": "text", "text": f"Frame {frame_idx}: {prompt}"},
                ],
            }
        ]

        result = client.chat_completion(
            messages=messages,
            model=VLM_MODEL,
            max_tokens=128,
        )

        return result.choices[0].message.content

    except Exception as exc:
        return f"[VLM Error: {type(exc).__name__}: {str(exc)[:100]}]"


def _aggregate_descriptions(frame_descriptions: list, original_prompt: str) -> str:
    """Aggregate frame descriptions using an LLM."""
    try:
        descriptions_text = "\n".join(
            f"Frame {d['frame_index']} (t={d['timestamp_seconds']}s): {d['description']}"
            for d in frame_descriptions
        )

        prompt = f"""Based on these frame-by-frame descriptions of a video, provide a coherent overall description.

Original question: {original_prompt}

Frame descriptions:
{descriptions_text}

Overall description:"""

        result = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=LLM_MODEL,
            max_tokens=256,
            temperature=0.7,
        )
        return result.choices[0].message.content
        return result

    except Exception as exc:
        return f"[Aggregation Error: {type(exc).__name__}: {str(exc)[:100]}]"


# ------------------------------------------------------------------
# Run with: uvicorn plm_fastapi:app --host 0.0.0.0 --port 8000
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

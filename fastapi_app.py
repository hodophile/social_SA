"""
FastAPI server for side-by-side sentiment/emotion comparison.

Endpoints:
  POST /analyze       → run both pipelines and return JSON
  GET  /              → serve static HTML UI
  GET  /health        → health check
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# CPU-only Space fix: neutralise spaces.GPU before Melisa loads it
import spaces
spaces.GPU = lambda **kwargs: lambda f: f

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent / "melisa_poc"))

from melisa_bridge import analyze_with_melisa
from llm_fusion_pipeline import TimestampLLMFusionPipeline

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# ------------------------------------------------------------------
# Lazy singleton
# ------------------------------------------------------------------
_llm_pipeline: Optional[TimestampLLMFusionPipeline] = None


def get_llm_pipeline() -> TimestampLLMFusionPipeline:
    global _llm_pipeline
    if _llm_pipeline is None:
        _llm_pipeline = TimestampLLMFusionPipeline(
            openrouter_api_key=OPENROUTER_API_KEY,
            openrouter_model=OPENROUTER_MODEL,
        )
    return _llm_pipeline


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="SA Social Media Sentiment Analysis", version="1.0.0")

# Serve static files (the HTML UI)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze")
def analyze(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    sample_fps: float = Form(1.0),
):
    """Run both Melisa POC and Timestamp-LLM-Fusion pipelines."""
    # Save uploaded files temporarily
    tmp_dir = Path("/tmp/sa_uploads")
    tmp_dir.mkdir(exist_ok=True)

    image_path: Optional[str] = None
    video_path: Optional[str] = None

    if image and image.filename:
        image_path = str(tmp_dir / image.filename)
        with open(image_path, "wb") as f:
            f.write(image.file.read())

    if video and video.filename:
        video_path = str(tmp_dir / video.filename)
        with open(video_path, "wb") as f:
            f.write(video.file.read())

    # Determine media path for Melisa (video wins over image)
    media_path = video_path or image_path

    # ---------------- Melisa POC ----------------
    melisa_res = analyze_with_melisa(
        text=None if media_path else (text or None),
        media_path=media_path,
    )

    # ---------------- Timestamp-LLM ----------------
    llm_res = get_llm_pipeline().analyze(
        text=text or None,
        image_path=Path(image_path) if image_path else None,
        video_path=Path(video_path) if video_path else None,
        sample_fps=sample_fps,
    )

    # ---------------- Comparison ----------------
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
            "primary_emotion_score": llm_emo.get("primary_emotion_score"),
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

    # Cleanup temp files
    for p in (image_path, video_path):
        if p:
            try:
                os.remove(p)
            except Exception:
                pass

    return {
        "melisa": melisa_res,
        "timestamp_llm": llm_res,
        "comparison": comparison,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the simple HTML UI."""
    html_file = static_dir / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>UI not found. Place static/index.html</h1>"


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

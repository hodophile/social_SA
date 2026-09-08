"""
FastAPI server with separate pages and APIs for:
  - Initial Setup (Melisa POC)
  - Latest (Timestamp-LLM-Fusion)
  - Compare (side-by-side)

Pages:
  GET /              → landing page with links
  GET /initial       → Melisa POC UI
  GET /latest        → Timestamp-LLM UI
  GET /compare       → side-by-side comparison UI

APIs:
  POST /api/initial/analyze   → Melisa POC only
  POST /api/latest/analyze    → Timestamp-LLM only
  POST /api/compare/analyze   → both side-by-side
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
app = FastAPI(title="SA Social Media Sentiment Analysis", version="2.0.0")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _save_upload(file: Optional[UploadFile], tmp_dir: Path) -> Optional[str]:
    if not file or not file.filename:
        return None
    path = str(tmp_dir / file.filename)
    with open(path, "wb") as f:
        f.write(file.file.read())
    return path


def _cleanup(*paths: Optional[str]):
    for p in paths:
        if p:
            try:
                os.remove(p)
            except Exception:
                pass


# ==================================================================
# HEALTH
# ==================================================================
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "2.0.0"}


# ==================================================================
# PAGES
# ==================================================================
@app.get("/", response_class=HTMLResponse)
def index():
    html = (static_dir / "index.html").read_text(encoding="utf-8") if (static_dir / "index.html").exists() else None
    if html:
        return html
    return "<h1>Place static/index.html</h1>"


@app.get("/initial", response_class=HTMLResponse)
def initial_page():
    html_file = static_dir / "initial.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Initial Setup UI not found</h1>"


@app.get("/latest", response_class=HTMLResponse)
def latest_page():
    html_file = static_dir / "latest.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Latest Setup UI not found</h1>"


@app.get("/compare", response_class=HTMLResponse)
def compare_page():
    html_file = static_dir / "compare.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Compare UI not found</h1>"


# ==================================================================
# API — INITIAL SETUP (Melisa POC only)
# ==================================================================
@app.post("/api/initial/analyze")
def api_initial_analyze(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
):
    """Run ONLY the Melisa POC pipeline."""
    tmp_dir = Path("/tmp/sa_uploads")
    tmp_dir.mkdir(exist_ok=True)

    image_path = _save_upload(image, tmp_dir)
    video_path = _save_upload(video, tmp_dir)
    media_path = video_path or image_path

    try:
        result = analyze_with_melisa(
            text=None if media_path else (text or None),
            media_path=media_path,
        )
        return {"pipeline": "melisa_poc", "result": result}
    finally:
        _cleanup(image_path, video_path)


# ==================================================================
# API — LATEST (Timestamp-LLM-Fusion only)
# ==================================================================
@app.post("/api/latest/analyze")
def api_latest_analyze(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    sample_fps: float = Form(1.0),
):
    """Run ONLY the Timestamp-LLM-Fusion pipeline."""
    tmp_dir = Path("/tmp/sa_uploads")
    tmp_dir.mkdir(exist_ok=True)

    image_path = _save_upload(image, tmp_dir)
    video_path = _save_upload(video, tmp_dir)

    try:
        result = get_llm_pipeline().analyze(
            text=text or None,
            image_path=Path(image_path) if image_path else None,
            video_path=Path(video_path) if video_path else None,
            sample_fps=sample_fps,
        )
        return {"pipeline": "timestamp_llm_fusion", "result": result}
    finally:
        _cleanup(image_path, video_path)


# ==================================================================
# API — COMPARE (both side-by-side)
# ==================================================================
@app.post("/api/compare/analyze")
def api_compare_analyze(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    sample_fps: float = Form(1.0),
):
    """Run BOTH pipelines and return comparison."""
    tmp_dir = Path("/tmp/sa_uploads")
    tmp_dir.mkdir(exist_ok=True)

    image_path = _save_upload(image, tmp_dir)
    video_path = _save_upload(video, tmp_dir)
    media_path = video_path or image_path

    try:
        # Melisa
        melisa_res = analyze_with_melisa(
            text=None if media_path else (text or None),
            media_path=media_path,
        )

        # Timestamp-LLM
        llm_res = get_llm_pipeline().analyze(
            text=text or None,
            image_path=Path(image_path) if image_path else None,
            video_path=Path(video_path) if video_path else None,
            sample_fps=sample_fps,
        )

        # Comparison
        llm_emo = llm_res.get("emotion_analysis", {})
        comparison = {
            "initial_version": {
                "matched_emotion": melisa_res.get("matched_emotion"),
                "initial_label": melisa_res.get("initial_label"),
                "valence": melisa_res.get("valence"),
                "confidence": melisa_res.get("initial_confidence"),
            },
            "latest_version": {
                "primary_emotion": llm_emo.get("primary_emotion"),
                "primary_emotion_score": llm_emo.get("primary_emotion_score"),
                "valence": llm_emo.get("valence"),
                "confidence": llm_emo.get("confidence"),
            },
        }
        v1_emo = comparison["initial_version"]["matched_emotion"]
        v2_emo = comparison["latest_version"]["primary_emotion"]
        if v1_emo and v2_emo:
            comparison["emotions_match"] = v1_emo == v2_emo
            v1_val = comparison["initial_version"]["valence"]
            v2_val = comparison["latest_version"]["valence"]
            if v1_val is not None and v2_val is not None:
                comparison["valence_gap"] = round(abs(v1_val - v2_val), 3)

        return {
            "melisa": melisa_res,
            "timestamp_llm": llm_res,
            "comparison": comparison,
        }
    finally:
        _cleanup(image_path, video_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

# Temporal‑Aware 8‑Emotion Video Sentiment (Melisa‑based)

This folder implements a **drop‑in replacement** for Melisa POC’s video analyzer that:

* Extracts **8‑emotion distributions** (Plutchik) from each sampled frame using a lightweight CNN (ResNet‑18 → FER2013 → mapped to joy, sadness, anger, fear, surprise, disgust, trust, anticipation).
* Aggregates frame emotions with an **exponential‑decay temporal weighting** (more weight to recent frames) – giving the model a sense of motion / trend.
* Uses the **same audio pipeline** (Whisper → RoBERTa) **or** an OpenRouter emotion call on the transcript for an audio‑based emotion distribution.
* Fuses the three modalities (caption, video, audio) with Melisa’s existing **confidence‑weighted late fusion** (`fuse_modalities`), yielding a single 8‑emotion distribution, valence, and confidence.
* Returns JSON in the **exact format** used by the current Qwen + OpenRouter pipeline so it can be swapped into the existing Gradio/Streamlit front‑end without any UI changes.

## Usage
```python
from melisa_temporal_emotion.pipeline import TemporalEmotionPipeline
from pathlib import Path

pipe = TemporalEmotionPipeline(openrouter_api_key="sk-or-v1-...")   # set to None for offline heuristics
result = pipe.analyze(
    text="I am so excited about this new project!",
    video_path=Path("/path/to/video.mp4"),
    sample_fps=1.0,
)

import json
print(json.dumps(result, indent=2))
```

## Dependencies
See `requirements.txt`.  Install with:
```bash
pip install -r requirements.txt
```

## Notes
* The video pipeline still extracts OCR and runs the original Melisa visual sentiment analyzer (SigLIP) – we keep those for compatibility and simply ignore their output in favor of the emotion net.
* If you do **not** provide an `OPENROUTER_API_KEY`, the audio branch falls back to the RoBERTa‑based sentiment (heuristic mapping to 8 emotions) – the pipeline will still work, just without the LLM‑based audio emotion.
* The temporal half‑life (`half_life_seconds=2.0` in `video_analyzer.py`) can be tuned to make the model more or less sensitive to recent frames.

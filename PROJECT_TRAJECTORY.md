# Project Trajectory — SA Social Media Sentiment Analysis

## 1. Melisa POC (Initial Setup)
**What:** Vendored pipeline from `ajayck007/myuni-sentiment-poc-melisa-v1`  
**How it works:**
- Text → RoBERTa sentiment
- Image/video frames → SigLIP zero-shot visual sentiment (positive/neutral/negative)
- Audio → Whisper ASR → RoBERTa sentiment on transcript
- Late fusion: confidence-weighted average of modalities

**Output:** 3-class label (positive/neutral/negative) + score + confidence  
**Status:** ✅ Working. Handles text, image, video.  
**Remarks:**
- Frame-level scoring is independent (no temporal model)
- Uses `@spaces.GPU` decorator for SigLIP forward — crashes on CPU-only Spaces
- Fixed by monkey-patching `spaces.GPU` to no-op when no GPU available

---

## 2. Qwen2.5-VLM-3B Setup
**What:** Custom multimodal pipeline using Qwen2.5-VLM-3B-Instruct  
**How it works:**
- Text/image/video → Qwen VLM generates structured JSON
- JSON parsed into: understanding, risk, action, emotion_analysis
- Fallback to mock mode (`QWEN_MOCK=1`) for local testing without GPU

**Output:** Full JSON with execution_time_seconds, understanding, risk, action, emotion_analysis  
**Status:** ⚠️ Partially working. Text-only reliable; video/image heavy on GPU.  
**Remarks:**
- Required GPU for real inference; mock mode for local dev
- Gradio UI had side-by-side comparison with Melisa POC
- Replaced by Temporal Emotion pipeline to avoid GPU dependency

---

## 3. Temporal Emotion (Current Setup)
**What:** New 8-emotion pipeline with temporal awareness  
**How it works:**
- Video frames → per-frame ResNet-18 CNN (FER2013-based) → 8 Plutchik emotions
- Temporal aggregation: exponential-decay weighted average across frame sequence
- Caption → RoBERTa sentiment → heuristic mapping to 8 emotions
- Audio transcript → OpenRouter API (or RoBERTa fallback) → 8 emotions
- Late fusion: confidence-weighted fusion of 8-emotion distributions

**Output:** Same JSON format as Qwen (execution_time_seconds, understanding, risk, action, emotion_analysis)  
**Status:** ✅ Working after fixes.  
**Remarks:**
- EmotionNet initially failed on numpy arrays (expected PIL Image) — fixed
- Audio analysis fails on video-only files (no audio stream) — expected, non-critical
- Runs on CPU — no GPU required
- Side-by-side comparison with Melisa POC in Gradio UI

---

## Summary Table

| Setup | Modality | Emotions | GPU Needed | Status |
|-------|----------|----------|------------|--------|
| Melisa POC | text/image/video | 3 (pos/neu/neg) | Optional (SigLIP uses GPU if avail) | ✅ Working |
| Qwen2.5-VLM-3B | text/image/video | 8 (full) | Yes (or mock) | ⚠️ Replaced |
| Temporal Emotion | text/image/video | 8 (full) | No | ✅ Working |

# Timestamp-LLM-Fusion Architecture

## Overview

The Timestamp-LLM-Fusion pipeline sends **per-frame SigLIP emotions with timestamps** + **audio transcript** + **caption text** to an LLM (OpenRouter GPT-4o-mini) for final 8-emotion synthesis. The LLM sees the full temporal context and can reason about emotion shifts, cross-modal agreement/disagreement, and sarcasm/irony.

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Video + Caption                          │
│                    e.g., 5-second video + "I love this!"                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: FRAME EXTRACTION (per_frame_siglip.py)                        │
│                                                                         │
│  Video ──► ffmpeg extracts frames at sample_fps (e.g., 1 fps)          │
│                                                                         │
│  5s video @ 1 fps = 5 frames                                           │
│                                                                         │
│  Output:                                                                │
│    Frame 0 @ t=0.0s  →  /tmp/frame_000.jpg                             │
│    Frame 1 @ t=1.0s  →  /tmp/frame_001.jpg                             │
│    Frame 2 @ t=2.0s  →  /tmp/frame_002.jpg                             │
│    Frame 3 @ t=3.0s  →  /tmp/frame_003.jpg                             │
│    Frame 4 @ t=4.0s  →  /tmp/frame_004.jpg                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: PER-FRAME SIGLIP EMOTION CLASSIFICATION                        │
│  (SigLIPFrameAnalyzer._classify_frame)                                 │
│                                                                         │
│  Each frame → SigLIP model compares image to 8 text prompts:           │
│                                                                         │
│    "a joyful, happy, delighted person smiling"         → joy score      │
│    "a sad, crying, sorrowful person looking down"      → sadness score  │
│    "an angry, furious, enraged person shouting"        → anger score    │
│    "a fearful, scared, terrified person trembling"     → fear score     │
│    "a surprised, astonished, shocked person..."        → surprise score │
│    "a disgusted, revolted, nauseated person grimacing" → disgust score  │
│    "a trusting, calm, peaceful, relaxed person"        → trust score    │
│    "an excited, eager, anticipating person..."         → anticipation   │
│                                                                         │
│  SigLIP returns logits → softmax → probability distribution             │
│                                                                         │
│  Output per frame:                                                      │
│    t=0.0s: {joy: 0.15, sadness: 0.05, anger: 0.60, fear: 0.02, ...}   │
│    t=1.0s: {joy: 0.20, sadness: 0.10, anger: 0.50, fear: 0.03, ...}   │
│    t=2.0s: {joy: 0.25, sadness: 0.15, anger: 0.40, fear: 0.05, ...}   │
│    ...                                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: AUDIO TRANSCRIPTION (parallel)                                │
│  (AudioAnalyzer from melisa_poc)                                       │
│                                                                         │
│  Video ──► ffmpeg extracts audio ──► Faster-Whisper ASR                │
│                                                                         │
│  Output:                                                                │
│    transcript: "I am so angry right now, I can't believe this!"        │
│    (or None if no audio stream)                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: BUILD LLM PROMPT (llm_fusion_pipeline.py)                     │
│                                                                         │
│  Combines ALL evidence into one structured prompt:                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  You are an expert multimodal emotion analyst.                  │   │
│  │                                                                 │   │
│  │  === PER-FRAME VISUAL EVIDENCE ===                              │   │
│  │    t= 0.00s: joy=0.150, sadness=0.050, anger=0.600, ...        │   │
│  │    t= 1.00s: joy=0.200, sadness=0.100, anger=0.500, ...        │   │
│  │    t= 2.00s: joy=0.250, sadness=0.150, anger=0.400, ...        │   │
│  │    t= 3.00s: joy=0.300, sadness=0.200, anger=0.300, ...        │   │
│  │    t= 4.00s: joy=0.350, sadness=0.250, anger=0.200, ...        │   │
│  │                                                                 │   │
│  │  === AUDIO TRANSCRIPT ===                                       │   │
│  │    I am so angry right now, I can't believe this!               │   │
│  │                                                                 │   │
│  │  === TEXT CAPTION ===                                           │   │
│  │    I love this!                                                 │   │
│  │                                                                 │   │
│  │  === INSTRUCTIONS ===                                           │   │
│  │  Based on ALL evidence, produce JSON with:                      │   │
│  │    - primary_emotion, secondary_emotion                         │   │
│  │    - emotion_scores (8 emotions, sum to 1.0)                   │   │
│  │    - valence (-1.0 to +1.0), confidence (0.0-1.0)              │   │
│  │    - rationale explaining how you combined all sources          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Key insight: LLM sees TEMPORAL DYNAMICS                               │
│    - Anger starts high (0.60) at t=0s                                  │
│    - Anger decreases over time (0.20 at t=4s)                          │
│    - Joy increases over time (0.15 → 0.35)                             │
│    - Caption says "I love this!" (positive)                            │
│    - Audio says "I am so angry" (negative)                             │
│                                                                         │
│  LLM can reason: "Visual shows anger cooling down over time,           │
│  but audio reveals genuine anger. Caption is sarcastic/ironic."        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 5: LLM SYNTHESIS (OpenRouter GPT-4o-mini)                        │
│  (_call_llm)                                                           │
│                                                                         │
│  POST https://openrouter.ai/api/v1/chat/completions                    │
│                                                                         │
│  LLM processes the full context and returns JSON:                      │
│                                                                         │
│  {                                                                      │
│    "primary_emotion": "anger",                                          │
│    "secondary_emotion": "disgust",                                      │
│    "emotion_scores": {                                                  │
│      "joy": 0.05, "sadness": 0.10, "anger": 0.55,                     │
│      "fear": 0.05, "surprise": 0.10, "disgust": 0.10,                 │
│      "trust": 0.03, "anticipation": 0.02                               │
│    },                                                                   │
│    "valence": -0.65,                                                    │
│    "confidence": 0.85,                                                  │
│    "rationale": "Audio reveals genuine anger despite caption's         │
│                   positive words. Visual shows anger dominant in        │
│                   early frames, suggesting the subject was genuinely    │
│                   upset when recording started."                        │
│  }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 6: OUTPUT STRUCTURE                                              │
│                                                                         │
│  {                                                                      │
│    "execution_time_seconds": 3.45,                                      │
│    "post_id": "timestamp-llm-post",                                     │
│    "understanding": {                                                   │
│      "content": "I love this!",                                         │
│      "tone": "anger",                                                   │
│      "visual_description": "5 frames analyzed with timestamps",         │
│      "audio_description": "Audio transcribed: I am so angry..."         │
│    },                                                                   │
│    "emotion_analysis": {                                                │
│      "emotion_model": "timestamp-llm-fusion (openai/gpt-4o-mini)",     │
│      "primary_emotion": "anger",                                        │
│      "secondary_emotion": "disgust",                                    │
│      "emotion_scores": {...},                                           │
│      "valence": -0.65,                                                  │
│      "confidence": 0.85,                                                │
│      "rationale": "Audio reveals genuine anger..."                     │
│    },                                                                   │
│    "frame_timestamps": [  ←── RAW DATA FOR TRANSPARENCY                 │
│      {"timestamp_seconds": 0, "emotions": {...}},                       │
│      {"timestamp_seconds": 1, "emotions": {...}},                       │
│      ...                                                                │
│    ]                                                                    │
│  }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘

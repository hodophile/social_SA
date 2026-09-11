---
title: VideoPrism Video Analysis
emoji: 🎬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# VideoPrism Video Analysis

This Hugging Face Space runs `google/videoprism-base-f16r288` locally for video understanding.

## How it works

1. Upload a video
2. Enter an optional custom prompt/question
3. The Space extracts video features using VideoPrism
4. Performs zero-shot classification for:
   - **Content category** (sports, cooking, music, travel, etc.)
   - **Emotion detection** (joy, sadness, anger, fear, surprise, etc.)
   - **Custom prompt matching** (yes/no classification for your question)
5. Returns a structured explanation with confidence scores

## Model

- **Video encoder:** `google/videoprism-base-f16r288` (114M params)
- **Text encoder:** `sentence-transformers/all-MiniLM-L6-v2`
- **Approach:** Zero-shot classification via cosine similarity between video and text embeddings

## Hardware

This Space works on both CPU and GPU tiers. GPU is recommended for faster inference.

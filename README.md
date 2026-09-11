---
title: PLM Video Analysis
emoji: 🎥
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# PLM Video Analysis (Local Model)

This Hugging Face Space runs `facebook/Perception-LM-1B` locally for video understanding.

## How it works

1. Upload a video
2. Enter a prompt/question
3. The Space downloads and runs `facebook/Perception-LM-1B` locally
4. Uses `decord` video backend and 32-frame sampling
5. Generates a response directly from the model

## Hardware

This Space requires a **GPU tier** because the model runs locally on CUDA.

## Required Secrets

No secrets required for public models. Ensure the Space has:

- Sufficient disk space for the 1B model download
- GPU hardware tier

## Implementation

This implementation mirrors `tests/plm2.py` but accepts user-uploaded videos instead of downloading from a dataset.

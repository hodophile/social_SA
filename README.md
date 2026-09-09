# PLM Video Analysis (HF Inference)

This Hugging Face Space demonstrates video understanding using Hugging Face Inference API instead of local model loading.

## How it works

1. Upload a video
2. Enter a prompt/question
3. The Space extracts frames using ffmpeg
4. Each frame is sent to a Vision-Language Model (Qwen2.5-VL-3B-Instruct) via HF Inference API
5. Frame descriptions are aggregated by an LLM (Llama-3.2-3B-Instruct)

## Required Secrets

Set in Space Settings → Secrets:

- `HF_TOKEN` — your Hugging Face access token

## Models Used

- VLM: `Qwen/Qwen2.5-VL-3B-Instruct`
- LLM: `meta-llama/Llama-3.2-3B-Instruct`
- PLM: `facebook/Perception-LM-1B` (for text-only if available)

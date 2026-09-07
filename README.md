---
title: Social Media Post Analysis (Qwen2.5-VLM-3B)
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# Social Media Post Analysis using Qwen2.5-VLM-3B

End-to-end multimodal pipeline to analyze posts (video / image / text)
and provide structured understanding and risk scoring.

**Pipeline:** modality preprocessing → Qwen2.5-VLM-3B → structured JSON
(content, tone, intent, entities) → harm classifier → per-post risk
score → downstream action (`ALLOW` / `REVIEW` / `BLOCK`).

## Usage

Provide any combination of text, image, and video, then submit.
The output is the full analysis JSON:

```json
{
  "post_id": "space-post",
  "understanding": { "content": "...", "tone": "...", "potential_harm": [] },
  "risk": { "score": 0.0, "labels": [] },
  "action": "ALLOW"
}
```

## Configuration (Space secrets / env vars)

| Variable    | Default                     | Description                  |
|-------------|-----------------------------|------------------------------|
| `QWEN_MOCK`  | `0`                         | `1` uses the mock (CPU ok)   |
| `QWEN_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | HF model id or local path  |
| `OPENROUTER_API_KEY` | —                   | enables emotion analysis     |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | emotion model on OpenRouter  |

## Comparative deployment: Melisa POC

The Space also runs the vendored Melisa sentiment POC
(`melisa_poc/`, from
[ajayck007/myuni-sentiment-poc-melisa-v1](https://huggingface.co/spaces/ajayck007/myuni-sentiment-poc-melisa-v1))
side by side with our pipeline on the same input.

Their fusion (`melisa_poc/src/fusion.py`, `config/fusion.yaml`):
confidence-weighted late fusion
`fused = sum(score_i * weight_i * conf_i) / sum(weight_i * conf_i)`,
thresholds +0.15 / -0.15 -> positive / neutral / negative, with a
confidence penalty on modality conflict.

Final labels are matched onto our emotion schema
(`melisa_bridge.py`): positive -> joy, neutral -> trust,
negative -> sadness. Their fused score doubles as valence on the
same -1..+1 scale, so the UI can show `emotions_match` and
`valence_gap` between the two pipelines.

## OpenRouter emotion analysis

When `OPENROUTER_API_KEY` is set, an LLM reads the analyzer's
structured understanding (content, tone, intent, visual/audio
descriptions) and identifies the **emotion of the content**:

- `primary_emotion` / `secondary_emotion` (Plutchik's 8 basic emotions)
- `emotion_scores` — raw 0–1 intensity per emotion
- `emotion_distribution` — normalized scores (sum = 1)
- `valence` — overall negativity (−1) to positivity (+1)
- `primary_emotion_score` — headline metric:
  `distribution[primary] × confidence`

> The default real-model inference requires a GPU hardware tier.
> Set `QWEN_MOCK=1` to run the placeholder pipeline on the free
> CPU tier.

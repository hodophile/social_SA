# Emotion Analysis Metrics

Reference for the metrics produced by the OpenRouter emotion-analysis
step in `app.py` (`analyze_emotion()`), shown in the Gradio UI panel
"Emotion analysis (OpenRouter)".

## Output fields

`primary_emotion` -> string label -> which emotion won. One of
Plutchik's 8 basic emotions: joy, sadness, anger, fear, surprise,
disgust, trust, anticipation.

`secondary_emotion` -> string or null -> the runner-up emotion, if the
model detects a blend.

`emotion_scores` -> dict of 0-1 values -> raw intensity per emotion,
as judged by the model.

`emotion_distribution` -> dict, sums to 1 -> emotion_scores normalized
into a probability-like distribution.

`valence` -> -1 to +1 -> overall pleasantness of the content (see
below).

`confidence` -> 0-1 -> the model's self-reported certainty in its
judgment.

`primary_emotion_score` -> 0-1 -> headline metric: strength and
reliability of the primary emotion (see below).

`rationale` -> string -> one-sentence explanation from the model.

## primary_emotion vs primary_emotion_score

The two fields answer different questions:

`primary_emotion` -> the classification -> which emotion is dominant.

`primary_emotion_score` -> how strongly to believe it:

    primary_emotion_score = emotion_distribution[primary_emotion] x confidence

It combines two independent signals:

1. `emotion_distribution[primary]` -> relative dominance -> how much
   the emotion stands out from the others. A post that is 90% joy is
   very different from one that is 30% joy / 25% sadness / 20% fear,
   even though "joy" wins both times.

2. `confidence` -> signal clarity -> how clear the emotional signal
   was (unambiguous happy caption vs. vague or mixed content).

Why multiply? Each alone is misleading:

- high distribution + low confidence -> winner picked, but model
  unsure (ambiguous or sarcastic content) -> product stays low
- low distribution + high confidence -> model certain the post is
  emotionally mixed -> product stays low
- high distribution + high confidence -> clear, dominant emotion ->
  product near 1.0

Typical usage: display the label; threshold on the score (e.g., only
surface the emotion when primary_emotion_score >= 0.5).

Caveat: confidence is self-reported by the LLM, not calibrated. LLMs
often report 0.9+ even when wrong. A calibrated alternative is an
entropy-based confidence derived from emotion_distribution.

## Valence

Valence is the overall pleasantness of the emotion on a single -1 to
+1 scale:

    -1.0 ---------------- 0 ---------------- +1.0
     very negative      neutral          very positive
     (anger, fear,     (calm, factual,   (joy, trust,
      sadness, disgust)  mixed emotions)   excitement)

How it relates to the other fields:

- emotion_scores -> multi-dimensional -> 8 separate intensities.
- valence -> collapses that into one number -> "overall, does this
  content feel good or bad?"

Examples:

- "I got the job! So thrilled!" -> primary: joy -> valence: +0.9
- "They killed the project we loved." -> primary: sadness -> valence: -0.7
- "I can't believe he did that" (angry) -> primary: anger -> valence: -0.8
- "Wait... what just happened?!" -> primary: surprise -> valence: ~0.0
  (surprise can go either way)
- "Meeting moved to 3pm." -> primary: neutral -> valence: ~0.0

Use valence for quick sorting or filtering ("show all strongly
negative posts"); use the emotion distribution for detail.

Note: the second classic dimension, arousal (calm vs. activated), is
not captured explicitly; the intensity in emotion_scores approximates
it.

## Example output

```json
{
  "emotion_model": "openai/gpt-4o-mini",
  "primary_emotion": "joy",
  "secondary_emotion": "anticipation",
  "emotion_scores": {
    "joy": 0.85, "sadness": 0.0, "anger": 0.0, "fear": 0.0,
    "surprise": 0.3, "disgust": 0.0, "trust": 0.4, "anticipation": 0.5
  },
  "emotion_distribution": {
    "joy": 0.415, "sadness": 0.0, "anger": 0.0, "fear": 0.0,
    "surprise": 0.146, "disgust": 0.0, "trust": 0.195, "anticipation": 0.244
  },
  "valence": 0.85,
  "confidence": 0.9,
  "primary_emotion_score": 0.373,
  "rationale": "The post expresses excitement and gratitude about a personal achievement."
}
```

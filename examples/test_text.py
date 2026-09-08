"""
test_text.py — one text-only post end-to-end.

Also demonstrates the harm-classifier path: a mock Qwen2.5-VLM-3B
output containing a harmful label must escalate the
downstream action to REVIEW/BLOCK.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from social_media_analyzer import (
    SocialMediaPost,
    QwenVLM,
    HarmClassifier,
    SocialMediaAnalyzer,
)

ASSETS = Path(__file__).parent / "assets"


class HarmfulMockQwenVLM(QwenVLM):
    """Mock Qwen2.5-VLM-3B that flags the post as harassment."""

    def generate(self, prompt, frames, audio_path, text):
        return {
            "content": text or "",
            "tone": "aggressive",
            "intent": "intimidation",
            "entities": [],
            "visual_description": "No visual content.",
            "audio_description": "No audio.",
            "potential_harm": ["harassment"],
        }


def main():
    text = (ASSETS / "sample_text.txt").read_text().strip()

    # --- benign text-only post ---
    post = SocialMediaPost(
        post_id="text-001",
        text=text,
    )

    analyzer = SocialMediaAnalyzer(
        vlm=QwenVLM(mock=True),
        classifier=HarmClassifier(),
    )

    result = analyzer.analyze(post)

    print("\n--- TEXT RESULT ---")
    print("normalized text:", repr(text))
    print("action:", result["action"])
    assert result["action"] == "ALLOW"

    # --- harmful text-only post ---
    harmful_post = SocialMediaPost(
        post_id="text-002",
        text="some threatening message",
    )

    analyzer.vlm = HarmfulMockQwenVLM(mock=True)
    result = analyzer.analyze(harmful_post)

    print("\n--- HARMFUL TEXT RESULT ---")
    print("risk:", result["risk"])
    print("action:", result["action"])
    assert result["action"] == "REVIEW"  # harassment weight 0.60

    print("text test PASSED")


if __name__ == "__main__":
    main()

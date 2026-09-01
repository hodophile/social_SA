"""
test_video.py — one video post end-to-end.

Runs: frame sampling + audio extraction -> PLM -> JSON
-> harm scoring -> downstream action.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from social_media_analyzer import (
    SocialMediaPost,
    PLM1B,
    HarmClassifier,
    SocialMediaAnalyzer,
)

ASSETS = Path(__file__).parent / "assets"


def main():
    post = SocialMediaPost(
        post_id="video-001",
        text="Check out this clip!",
        video_path=ASSETS / "sample_video.mp4",
    )

    analyzer = SocialMediaAnalyzer(
        plm=PLM1B(mock=True),  # set mock=False with --model for real inference
        classifier=HarmClassifier(),
    )

    result = analyzer.analyze(post, sample_fps=1.0)

    print("\n--- VIDEO RESULT ---")
    print("frames analyzed:",
          result["understanding"]["visual_description"])
    print("audio:", result["understanding"]["audio_description"])
    print("action:", result["action"])

    assert result["post_id"] == "video-001"
    assert result["action"] == "ALLOW"
    print("video test PASSED")


if __name__ == "__main__":
    main()

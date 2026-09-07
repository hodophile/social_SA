"""
test_image.py — one image post end-to-end.

The image is loaded as a single 'frame' and passed to Qwen2.5-VLM-3B.
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


def main():
    post = SocialMediaPost(
        post_id="image-001",
        image_path=ASSETS / "sample_image.png",
    )

    analyzer = SocialMediaAnalyzer(
        vlm=QwenVLM(mock=False),
        classifier=HarmClassifier(),
    )

    result = analyzer.analyze(post)

    import pdb;pdb.set_trace()

    print("\n--- IMAGE RESULT ---")
    print("visual:", result["understanding"]["visual_description"])
    print("action:", result["action"])

    assert "1 visual frames" in (
        result["understanding"]["visual_description"]
    )
    assert result["action"] == "ALLOW"
    print("image test PASSED")


if __name__ == "__main__":
    main()

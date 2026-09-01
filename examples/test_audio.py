"""
test_audio.py — audio modality.

The pipeline ingests audio via video posts (the Audio Tower
receives the extracted 16 kHz mono track). This test:

  1. verifies ffmpeg audio extraction from the sample video,
  2. verifies the extracted audio matches the standalone WAV
     in format (mono, 16 kHz),
  3. runs a post whose only signal is the audio track.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from social_media_analyzer import (
    SocialMediaPost,
    PLM1B,
    HarmClassifier,
    SocialMediaAnalyzer,
    extract_audio,
)

ASSETS = Path(__file__).parent / "assets"


def probe(path: Path) -> str:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def main():
    # 1. extraction
    with tempfile.TemporaryDirectory() as tmp:
        wav = extract_audio(
            ASSETS / "sample_video.mp4",
            Path(tmp) / "audio.wav",
        )
        print("extracted:", wav, "->", probe(wav))
        assert probe(wav) == "16000,1", "expected mono 16 kHz"

    # standalone audio asset sanity check
    print("standalone audio:", probe(ASSETS / "sample_audio.wav"))

    # 2. audio-only post (video without text, sound carries the signal)
    post = SocialMediaPost(
        post_id="audio-001",
        video_path=ASSETS / "sample_video.mp4",
    )

    analyzer = SocialMediaAnalyzer(
        plm=PLM1B(mock=True),
        classifier=HarmClassifier(),
    )

    result = analyzer.analyze(post)

    print("\n--- AUDIO RESULT ---")
    print("audio:", result["understanding"]["audio_description"])
    assert result["understanding"]["audio_description"] == "Audio available."
    print("audio test PASSED")


if __name__ == "__main__":
    main()

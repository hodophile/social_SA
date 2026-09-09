"""
Test script for Perception-LM-1B (PLM) via Hugging Face Inference Client.

No local model download needed — just HF_TOKEN.

Usage:
    export HF_TOKEN=your_huggingface_token
    python tests/plm.py
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from huggingface_hub import InferenceClient

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN environment variable")

PLM_MODEL = "facebook/Perception-LM-1B"
VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

TEST_VIDEO = "/home/wsm/Downloads/test_vid_4.mp4"

# Initialize client
client = InferenceClient(token=HF_TOKEN)


def test_text_only():
    """Test text-only inference."""
    print("=" * 60)
    print("TEST 1: Text-only inference")
    print("=" * 60)

    try:
        result = client.chat_completion(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            model=PLM_MODEL,
            max_tokens=50,
        )
        print(f"Result: {result.choices[0].message.content}")
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}")
    print()


def test_image_inference():
    """Test image inference with VLM."""
    print("=" * 60)
    print("TEST 2: Image inference with VLM")
    print("=" * 60)

    # Create a simple test image or use an existing one
    test_image = "/tmp/test_image.png"

    # Try to extract one frame from video as test image
    if Path(TEST_VIDEO).exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", TEST_VIDEO, "-frames:v", "1", test_image],
            capture_output=True,
        )

    if not Path(test_image).exists():
        print("No test image available")
        return

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": f"file://{test_image}"},
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ]
        result = client.chat_completion(
            messages=messages,
            model=VLM_MODEL,
            max_tokens=100,
        )
        print(f"Result: {result.choices[0].message.content}")
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}")
    print()


def test_video_frames():
    """Extract frames and analyze with VLM."""
    print("=" * 60)
    print("TEST 3: Video frame extraction + VLM")
    print("=" * 60)

    if not Path(TEST_VIDEO).exists():
        print(f"Video not found: {TEST_VIDEO}")
        return

    # Extract 3 frames
    frames = extract_frames(TEST_VIDEO, num_frames=3)
    print(f"Extracted {len(frames)} frames")

    for i, frame_path in enumerate(frames):
        print(f"\n--- Frame {i} ---")
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": f"file://{frame_path}"},
                        {"type": "text", "text": "Describe what you see"},
                    ],
                }
            ]
            result = client.chat_completion(
                messages=messages,
                model=VLM_MODEL,
                max_tokens=100,
            )
            print(f"Description: {result.choices[0].message.content}")
        except Exception as exc:
            print(f"Error: {type(exc).__name__}: {exc}")

    # Cleanup
    for f in frames:
        try:
            os.unlink(f)
        except Exception:
            pass


def extract_frames(video_path: str, num_frames: int = 3) -> list:
    """Extract evenly spaced frames from video."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip()) if probe.returncode == 0 else 10.0

    frames = []
    for i in range(num_frames):
        timestamp = (i / max(num_frames - 1, 1)) * duration
        frame_path = f"/tmp/plm_test_frame_{i}.png"
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
             "-frames:v", "1", frame_path],
            capture_output=True,
        )
        if result.returncode == 0 and Path(frame_path).exists():
            frames.append(frame_path)

    return frames


if __name__ == "__main__":
    print("PLM Test via Hugging Face Inference Client")
    print(f"PLM Model: {PLM_MODEL}")
    print(f"VLM Model: {VLM_MODEL}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"HF_TOKEN set: {bool(HF_TOKEN)}")
    print()

    test_text_only()
    test_image_inference()
    test_video_frames()

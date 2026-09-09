"""
app.py – Gradio UI for PLM (Perception-LM-1B) video understanding.

Uses Hugging Face Inference Client instead of local model loading.
Supports video + text input.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
from huggingface_hub import InferenceClient

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
PLM_MODEL = os.environ.get("PLM_MODEL", "facebook/Perception-LM-1B")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else None

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def extract_frames(video_path: str, sample_fps: float = 1.0, max_frames: int = 8) -> list:
    """Extract frames from video using ffmpeg."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip()) if probe.returncode == 0 else 10.0

    num_frames = min(max_frames, int(duration * sample_fps))
    if num_frames < 1:
        num_frames = 1

    frames = []
    for i in range(num_frames):
        timestamp = (i / max(num_frames - 1, 1)) * duration
        frame_path = f"/tmp/plm_frame_{i:04d}.png"
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
             "-frames:v", "1", "-f", "image2", frame_path],
            capture_output=True,
        )
        if result.returncode == 0 and Path(frame_path).exists():
            frames.append(frame_path)

    return frames


def describe_frame(frame_path: str, prompt: str, frame_idx: int) -> str:
    """Send a single frame to VLM for description."""
    if not client:
        return "[Error: HF_TOKEN not set]"

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": f"file://{frame_path}"},
                    {"type": "text", "text": f"Frame {frame_idx}: {prompt}"},
                ],
            }
        ]

        result = client.chat_completion(
            messages=messages,
            model=VLM_MODEL,
            max_tokens=128,
        )

        return result.choices[0].message.content

    except Exception as exc:
        return f"[VLM Error: {type(exc).__name__}: {str(exc)[:100]}]"


def aggregate_descriptions(frame_descriptions: list, original_prompt: str) -> str:
    """Aggregate frame descriptions using an LLM."""
    if not client:
        return "[Error: HF_TOKEN not set]"

    try:
        descriptions_text = "\n".join(
            f"Frame {d['frame_index']} (t={d['timestamp_seconds']}s): {d['description']}"
            for d in frame_descriptions
        )

        prompt = f"""Based on these frame-by-frame descriptions of a video, provide a coherent overall description.

Original question: {original_prompt}

Frame descriptions:
{descriptions_text}

Overall description:"""

        result = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=LLM_MODEL,
            max_tokens=256,
            temperature=0.7,
        )
        return result.choices[0].message.content

    except Exception as exc:
        return f"[Aggregation Error: {type(exc).__name__}: {str(exc)[:100]}]"


# ------------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------------
def analyze_video(video_file, text_prompt, sample_fps, max_frames):
    """Analyze video with PLM via HF Inference Client."""
    if not client:
        return {"error": "HF_TOKEN not set. Set it in Space Settings → Secrets."}

    if video_file is None:
        return {"error": "Please upload a video file."}

    # Save uploaded video
    suffix = Path(video_file.name).suffix if hasattr(video_file, 'name') else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        if hasattr(video_file, 'read'):
            tmp.write(video_file.read())
        else:
            # Gradio file path
            import shutil
            shutil.copy(video_file, tmp.name)
        video_path = tmp.name

    try:
        # Extract frames
        frames = extract_frames(video_path, sample_fps, max_frames)
        if not frames:
            return {"error": "No frames could be extracted from video."}

        # Analyze each frame
        frame_descriptions = []
        for i, frame_path in enumerate(frames):
            desc = describe_frame(frame_path, text_prompt, i)
            frame_descriptions.append({
                "frame_index": i,
                "timestamp_seconds": round(i / sample_fps, 2) if sample_fps > 0 else i,
                "description": desc,
            })

        # Aggregate
        aggregated = aggregate_descriptions(frame_descriptions, text_prompt)

        return {
            "status": "ok",
            "vlm_model": VLM_MODEL,
            "llm_model": LLM_MODEL,
            "prompt": text_prompt,
            "num_frames": len(frames),
            "frame_descriptions": frame_descriptions,
            "aggregated_description": aggregated,
        }

    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)}"}
    finally:
        # Cleanup
        try:
            os.unlink(video_path)
            for f in frames:
                os.unlink(f)
        except Exception:
            pass


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------
with gr.Blocks(title="PLM Video Analysis (HF Inference)") as demo:
    gr.Markdown("# Perception-LM-1B Video Understanding")
    gr.Markdown("Uses Hugging Face Inference API — no local model download needed.")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            text_input = gr.Textbox(
                label="Prompt",
                value="Describe what happens in this video",
                lines=2,
            )
            sample_fps = gr.Slider(
                label="Frame Sampling (fps)",
                minimum=0.5,
                maximum=5.0,
                value=1.0,
                step=0.5,
            )
            max_frames = gr.Slider(
                label="Max Frames",
                minimum=1,
                maximum=16,
                value=8,
                step=1,
            )
            analyze_btn = gr.Button("Analyze", variant="primary")

        with gr.Column():
            output_json = gr.JSON(label="Result")
            output_text = gr.Textbox(label="Aggregated Description", lines=10)

    analyze_btn.click(
        fn=lambda v, t, s, m: (analyze_video(v, t, s, m), analyze_video(v, t, s, m).get("aggregated_description", "")),
        inputs=[video_input, text_input, sample_fps, max_frames],
        outputs=[output_json, output_text],
    )

if __name__ == "__main__":
    demo.launch()

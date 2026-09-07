"""
streamlit_app.py — local Streamlit UI for the analyzer.

Run:
    pip install streamlit opencv-python-headless numpy
    streamlit run streamlit_app.py

Defaults to the real Qwen2.5-VLM-3B (Qwen/Qwen2.5-VL-3B-Instruct).
For fast mock inference (no GPU):
    QWEN_MOCK=1 streamlit run streamlit_app.py
"""

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from social_media_analyzer import (
    SocialMediaPost,
    QwenVLM,
    HarmClassifier,
    SocialMediaAnalyzer,
)

st.set_page_config(
    page_title="Social Media Post Analysis",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def get_analyzer(mock: bool, model_path: str | None):
    return SocialMediaAnalyzer(
        vlm=QwenVLM(model_path=model_path, mock=mock),
        classifier=HarmClassifier(),
    )


def save_upload(uploaded, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix
    )
    tmp.write(uploaded.read())
    tmp.close()
    return Path(tmp.name)


# ---------------- sidebar ----------------

with st.sidebar:
    st.header("⚙️ Settings")

    mock_default = os.environ.get("QWEN_MOCK", "0") == "1"
    mock = st.toggle("Mock Qwen2.5-VLM-3B (no GPU needed)", value=mock_default)

    model_path = st.text_input(
        "Model (HF id or local path)",
        value=os.environ.get("QWEN_MODEL", QwenVLM.DEFAULT_MODEL),
        disabled=mock,
    )

    sample_fps = st.slider(
        "Video frame sampling (fps)", 0.2, 5.0, 1.0, 0.2
    )

    st.caption(
        "Pipeline: preprocess → Qwen2.5-VLM-3B → JSON → harm score → action"
    )

# ---------------- main ----------------

st.title("🛡️ Social Media Post Analysis")
st.caption("Multimodal pipeline using Qwen2.5-VLM-3B")

col1, col2 = st.columns(2)

with col1:
    text = st.text_area(
        "Post text / caption",
        placeholder="This is an example post caption about cats and dogs.",
        height=120,
    )

with col2:
    image = st.file_uploader(
        "Image (optional)", type=["png", "jpg", "jpeg", "webp"]
    )
    video = st.file_uploader(
        "Video (optional)", type=["mp4", "mov", "webm", "mkv"]
    )

if image:
    st.image(image, caption="Image input", width=320)
if video:
    st.video(video)

if st.button("Analyze", type="primary", use_container_width=True):

    if not text and not image and not video:
        st.warning("Provide at least one of: text, image, video.")
        st.stop()

    image_path = (
        save_upload(image, Path(image.name).suffix) if image else None
    )
    video_path = (
        save_upload(video, Path(video.name).suffix) if video else None
    )

    post = SocialMediaPost(
        post_id="streamlit-post",
        text=text or None,
        image_path=image_path,
        video_path=video_path,
    )

    analyzer = get_analyzer(
        mock=mock,
        model_path=None if mock else model_path,
    )

    with st.status("Running pipeline...", expanded=True) as status:
        try:
            result = analyzer.analyze(post, sample_fps=sample_fps)
            status.update(label="Pipeline complete", state="complete")
        except Exception as exc:
            status.update(label="Pipeline failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            st.stop()

    # ---------------- results ----------------

    action = result["action"]
    risk = result["risk"]
    understanding = result["understanding"]

    color = {"ALLOW": "green", "REVIEW": "orange", "BLOCK": "red"}[
        action
    ]

    m1, m2, m3 = st.columns(3)
    m1.metric("Action", action)
    m2.metric("Risk score", f"{risk['score']:.2f}")
    m3.metric(
        "Harm labels",
        ", ".join(risk["labels"]) or "none",
    )

    st.markdown(f"**Decision:** :{color}[{action}]")

    with st.expander("Understanding (Qwen2.5-VLM-3B output)", expanded=True):
        st.write(f"**Content:** {understanding['content']}")
        st.write(f"**Tone:** {understanding['tone']}  |  "
                 f"**Intent:** {understanding['intent']}")
        st.write(f"**Visual:** {understanding['visual_description']}")
        st.write(f"**Audio:** {understanding['audio_description']}")
        if understanding["entities"]:
            st.write(f"**Entities:** {', '.join(understanding['entities'])}")

    with st.expander("Full JSON"):
        st.json(result)

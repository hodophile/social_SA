"""
social_media_analyzer.py

Simple single-file pipeline:

    SocialMediaPost
          |
          v
    Modality preprocessing
          |
          +---- video -> sampled frames
          +---- video -> extracted audio
          +---- image -> image tensor
          +---- text  -> normalized text
          |
          v
       PLM-1B / PLM-AV
          |
          v
    Structured JSON
          |
          v
    Harm classifier
          |
          v
      Risk score
          |
          v
    Downstream action

The PLM class contains a mock implementation so the complete
pipeline can be tested immediately. Replace PLM1B.generate()
with the actual PLM-1B/PLM-AV inference call.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import argparse
import json
import subprocess
import tempfile


# ============================================================
# 1. RAW INPUT DATATYPE
# ============================================================

@dataclass
class SocialMediaPost:
    """
    Raw social-media post.

    A post may contain text, an image, a video, or any
    combination of these.
    """

    post_id: str
    text: Optional[str] = None
    image_path: Optional[Path] = None
    video_path: Optional[Path] = None


# ============================================================
# 2. PROCESSED INPUT DATATYPE
# ============================================================

@dataclass
class ProcessedPost:
    """
    Common representation passed to the multimodal model.

    Video:
        frames = sampled frames
        audio_path = extracted WAV

    Image:
        frames = [image]

    Text:
        text = normalized text
    """

    post_id: str
    text: Optional[str]
    frames: List[Any] = field(default_factory=list)
    audio_path: Optional[str] = None


# ============================================================
# 3. VIDEO PROCESSING
# ============================================================

def extract_video_frames(
    video_path: Path,
    sample_fps: float = 1.0,
) -> List[Any]:
    """
    Sample approximately sample_fps frames per second.

    Example:
        30 FPS video + sample_fps=1
        -> approximately 1 frame per second
    """

    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "OpenCV is required. Install with:\n"
            "pip install opencv-python"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if not video_fps or video_fps <= 0:
        video_fps = 30.0

    frame_interval = max(
        int(video_fps / sample_fps),
        1,
    )

    frames = []
    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % frame_interval == 0:
            frames.append(frame)

        frame_index += 1

    cap.release()

    return frames


# ============================================================
# 4. IMAGE PROCESSING
# ============================================================

def load_image(image_path: Path):
    """Load an image into memory."""

    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "OpenCV is required. Install with:\n"
            "pip install opencv-python"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Could not load image: {image_path}"
        )

    return image


# ============================================================
# 5. AUDIO PROCESSING
# ============================================================

def extract_audio(
    video_path: Path,
    output_path: Path,
) -> Path:
    """
    Extract mono 16-kHz audio from video using ffmpeg.
    """

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg and "
            "make sure it is available on PATH."
        )

    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Audio extraction failed:\n"
            + exc.stderr.decode(errors="ignore")
        )

    return output_path


# ============================================================
# 6. TEXT PROCESSING
# ============================================================

def normalize_text(
    text: Optional[str],
) -> Optional[str]:
    """Basic text normalization."""

    if text is None:
        return None

    return " ".join(text.strip().split())


# ============================================================
# 7. MODALITY PREPROCESSOR
# ============================================================

def preprocess_post(
    post: SocialMediaPost,
    sample_fps: float = 1.0,
    work_dir: Optional[Path] = None,
) -> ProcessedPost:
    """
    Convert raw post into the normalized multimodal format.
    """

    frames = []
    audio_path = None

    text = normalize_text(post.text)

    if work_dir is None:
        work_dir = Path(
            tempfile.mkdtemp(
                prefix=f"plm_{post.post_id}_"
            )
        )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # VIDEO
    # -------------------------

    if post.video_path:

        frames = extract_video_frames(
            post.video_path,
            sample_fps=sample_fps,
        )

        audio_file = work_dir / "audio.wav"

        extract_audio(
            post.video_path,
            audio_file,
        )

        audio_path = str(audio_file)

    # -------------------------
    # IMAGE
    # -------------------------

    elif post.image_path:

        frames = [
            load_image(post.image_path)
        ]

    return ProcessedPost(
        post_id=post.post_id,
        text=text,
        frames=frames,
        audio_path=audio_path,
    )


# ============================================================
# 8. PLM OUTPUT SCHEMA
# ============================================================

@dataclass
class PostUnderstanding:
    """
    Structured representation produced by PLM-1B.
    """

    content: str
    tone: str
    intent: str
    entities: List[str]

    visual_description: str
    audio_description: str

    potential_harm: List[str]

    def to_dict(self):
        return {
            "content": self.content,
            "tone": self.tone,
            "intent": self.intent,
            "entities": self.entities,
            "visual_description": self.visual_description,
            "audio_description": self.audio_description,
            "potential_harm": self.potential_harm,
        }


# ============================================================
# 9. PLM-1B / PLM-AV
# ============================================================

class PLM1B:
    """
    PLM-1B / PLM-AV wrapper.

    Architecture (see DESIGN.png):

        frames  -> Video Tower (PE-L + Temporal)   ~0.32B
        audio   -> Audio Tower (DAC-VAE + Encoder) ~1.1B
        text    -> Text Embedder (optional)
                        |
             Audio-Video Fusion (6-layer Transformer)
                        |
             PLM Decoder Head (generation/captioning)

    mock=True  -> deterministic placeholder output so the
                  full pipeline can be tested without a GPU
                  or checkpoint (used by the examples/).

    mock=False -> loads the real multimodal checkpoint via
                  HuggingFace transformers. All model-
                  specific code is isolated to this class.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(
        self,
        model_path: Optional[str] = None,
        mock: bool = True,
        device: Optional[str] = None,
    ):
        self.model_path = model_path or self.DEFAULT_MODEL
        self.mock = mock
        self.device = device

        if not mock:
            self._load_model()

    def _load_model(self):
        """
        Load the actual PLM-1B / PLM-AV checkpoint here.

        Keep model-specific code isolated to this class.

        Requires:
            pip install torch transformers accelerate
            pip install qwen-vl-utils   # for Qwen2.5-VL

        Swap AutoModelForImageTextToText / processor for the
        real PLM-1B (PLM-AV) towers + fusion checkpoint when
        it is available; the rest of the application must
        not change.
        """

        try:
            import torch
            from transformers import (
                AutoProcessor,
                AutoModelForImageTextToText,
            )
        except ImportError:
            raise RuntimeError(
                "Real model inference requires:\n"
                "pip install torch transformers accelerate"
            )

        self.device = self.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(
            f"      loading PLM from "
            f"{self.model_path} on {self.device}"
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map=self.device,
        )

        self.model.eval()

    def _run_real_inference(
        self,
        prompt: str,
        frames: List[Any],
        audio_path: Optional[str],
        text: Optional[str],
    ) -> Dict[str, Any]:
        """
        Single inference pass of the fused multimodal model.

        Frames (video samples or a single image) are passed
        as images. The audio waveform is fed to the Audio
        Tower when the checkpoint supports audio; for VLM-
        style checkpoints it is referenced in the prompt.
        """

        import json as _json

        try:
            import cv2
        except ImportError:
            raise RuntimeError(
                "OpenCV is required. Install with:\n"
                "pip install opencv-python"
            )

        content = []

        # ---- Video Tower input (sampled frames / image) ----
        for frame in frames[:8]:  # cap context length
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            content.append({"type": "image", "image": rgb})

        # ---- Audio Tower input ----
        audio_note = (
            f"An audio track was extracted from the video "
            f"({audio_path}). Account for it in "
            f"'audio_description'."
            if audio_path
            else "No audio track is present."
        )

        # ---- Text Embedder input ----
        user_text = (
            f"Post caption: {text}\n\n"
            if text
            else ""
        )

        content.append({
            "type": "text",
            "text": f"{prompt}\n{user_text}{audio_note}",
        })

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        import torch

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
            )

        generated = output_ids[
            :, inputs["input_ids"].shape[1]:
        ]

        raw = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]

        # Extract the JSON object from the response.
        start = raw.find("{")
        end = raw.rfind("}") + 1

        if start == -1 or end <= start:
            raise ValueError(
                "PLM did not return JSON:\n" + raw
            )

        return _json.loads(raw[start:end])

    def analyze(
        self,
        frames: List[Any],
        audio_path: Optional[str],
        text: Optional[str],
    ) -> Dict[str, Any]:

        prompt = """
Analyze this social media post.

Understand the visual, audio and textual content.

Return ONLY JSON:

{
    "content": "...",
    "tone": "...",
    "intent": "...",
    "entities": [],
    "visual_description": "...",
    "audio_description": "...",
    "potential_harm": []
}
"""

        return self.generate(
            prompt=prompt,
            frames=frames,
            audio_path=audio_path,
            text=text,
        )

    def generate(
        self,
        prompt: str,
        frames: List[Any],
        audio_path: Optional[str],
        text: Optional[str],
    ) -> Dict[str, Any]:

        if self.mock:

            # Deterministic placeholder output used by the
            # examples/ scripts so the complete pipeline
            # (preprocessing -> JSON -> scoring -> action)
            # can be tested without a GPU or checkpoint.

            return {
                "content": (
                    "Social media post containing "
                    "multimodal content."
                ),
                "tone": "neutral",
                "intent": "informational",
                "entities": [],
                "visual_description": (
                    f"{len(frames)} visual frames available."
                ),
                "audio_description": (
                    "Audio available."
                    if audio_path
                    else "No audio."
                ),
                "potential_harm": [],
            }

        return self._run_real_inference(
            prompt=prompt,
            frames=frames,
            audio_path=audio_path,
            text=text,
        )


# ============================================================
# 10. VALIDATE PLM OUTPUT
# ============================================================

def validate_understanding(
    data: Dict[str, Any],
) -> PostUnderstanding:

    required = [
        "content",
        "tone",
        "intent",
        "entities",
        "visual_description",
        "audio_description",
        "potential_harm",
    ]

    missing = [
        key for key in required
        if key not in data
    ]

    if missing:
        raise ValueError(
            f"PLM output is missing fields: {missing}"
        )

    return PostUnderstanding(
        content=data["content"],
        tone=data["tone"],
        intent=data["intent"],
        entities=data["entities"],
        visual_description=data["visual_description"],
        audio_description=data["audio_description"],
        potential_harm=data["potential_harm"],
    )


# ============================================================
# 11. HARM CLASSIFIER
# ============================================================

class HarmClassifier:
    """
    Very simple rule-based scoring implementation.

    Replace this later with a trained classifier.
    """

    HARM_WEIGHTS = {
        "violence": 0.90,
        "harassment": 0.60,
        "hate": 0.80,
        "sexual": 0.80,
        "self_harm": 0.90,
        "threat": 0.90,
        "abuse": 0.70,
    }

    def predict(
        self,
        understanding: PostUnderstanding,
    ) -> Dict[str, Any]:

        labels = understanding.potential_harm

        probabilities = {}

        for label in labels:
            probabilities[label] = (
                self.HARM_WEIGHTS.get(
                    label,
                    0.50,
                )
            )

        score = max(
            probabilities.values(),
            default=0.0,
        )

        return {
            "score": score,
            "labels": labels,
            "probabilities": probabilities,
        }


# ============================================================
# 12. DOWNSTREAM POLICY
# ============================================================

def decide_action(
    risk_result: Dict[str, Any],
) -> str:

    score = risk_result["score"]

    if score >= 0.80:
        return "BLOCK"

    if score >= 0.50:
        return "REVIEW"

    return "ALLOW"


# ============================================================
# 13. COMPLETE PIPELINE
# ============================================================

class SocialMediaAnalyzer:

    def __init__(
        self,
        plm: PLM1B,
        classifier: HarmClassifier,
    ):
        self.plm = plm
        self.classifier = classifier

    def analyze(
        self,
        post: SocialMediaPost,
        sample_fps: float = 1.0,
    ) -> Dict[str, Any]:

        # --------------------------------------------
        # STEP 1: RAW POST
        # --------------------------------------------

        print("[1/6] Received post")

        # --------------------------------------------
        # STEP 2: PREPROCESS
        # --------------------------------------------

        print("[2/6] Preprocessing modalities")

        processed = preprocess_post(
            post,
            sample_fps=sample_fps,
        )

        print(
            f"      frames = {len(processed.frames)}"
        )

        print(
            f"      audio  = "
            f"{processed.audio_path is not None}"
        )

        print(
            f"      text   = "
            f"{processed.text is not None}"
        )

        # --------------------------------------------
        # STEP 3: PLM-1B / PLM-AV
        # --------------------------------------------

        print("[3/6] Running PLM-1B / PLM-AV")

        plm_output = self.plm.analyze(
            frames=processed.frames,
            audio_path=processed.audio_path,
            text=processed.text,
        )

        # --------------------------------------------
        # STEP 4: STRUCTURED JSON
        # --------------------------------------------

        print("[4/6] Validating structured output")

        understanding = validate_understanding(
            plm_output
        )

        # --------------------------------------------
        # STEP 5: CLASSIFY / SCORE
        # --------------------------------------------

        print("[5/6] Classifying and scoring")

        risk = self.classifier.predict(
            understanding
        )

        # --------------------------------------------
        # STEP 6: DOWNSTREAM ACTION
        # --------------------------------------------

        action = decide_action(risk)

        print(
            f"[6/6] Downstream action = {action}"
        )

        return {
            "post_id": post.post_id,
            "understanding": understanding.to_dict(),
            "risk": risk,
            "action": action,
        }


# ============================================================
# 14. COMMAND LINE ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Social Media Post Analysis "
            "using PLM-1B / PLM-AV"
        )
    )

    parser.add_argument(
        "--post-id",
        default="post-001",
    )

    parser.add_argument(
        "--text",
        default=None,
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--model",
        default=None,
    )

    parser.add_argument(
        "--real-model",
        action="store_true",
        help=(
            "Use actual PLM instead of mock inference. "
            "Requires implementation of PLM1B._load_model()."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------
    # INPUT
    # --------------------------------------------

    post = SocialMediaPost(
        post_id=args.post_id,
        text=args.text,
        image_path=args.image,
        video_path=args.video,
    )

    # --------------------------------------------
    # MODELS
    # --------------------------------------------

    plm = PLM1B(
        model_path=args.model,
        mock=not args.real_model,
    )

    classifier = HarmClassifier()

    analyzer = SocialMediaAnalyzer(
        plm=plm,
        classifier=classifier,
    )

    # --------------------------------------------
    # RUN PIPELINE
    # --------------------------------------------

    result = analyzer.analyze(
        post,
        sample_fps=args.sample_fps,
    )

    # --------------------------------------------
    # RESULT
    # --------------------------------------------

    print("\n================ RESULT ================\n")

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

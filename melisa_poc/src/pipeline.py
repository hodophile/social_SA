"""Central MyUni sentiment analysis pipeline."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import spaces  # ZeroGPU patch before analyzer modules import torch

from src.analyzers.audio import AudioAnalyzer
from src.analyzers.image import ImageAnalyzer
from src.analyzers.text import TextSentimentAnalyzer
from src.analyzers.video import VideoAnalyzer
from src.config import (
    DEFAULT_FUSION,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VIDEO_SAMPLING,
    DEFAULT_VISUAL_MODEL,
    DEFAULT_WHISPER_MODEL,
    FusionConfig,
    VideoSamplingConfig,
)
from src.fusion import fuse_modalities
from src.media.ffmpeg_utils import FFmpegError, FFmpegNotFoundError
from src.routing.input_router import (
    CLIENT_ENABLED_MODALITIES,
    CapabilityStatus,
    InputRouter,
    InputType,
    RoutedAnalysisResult,
)
from src.runtime_info import build_poc_runtime_info
from src.schemas import (
    ActivityAnalysisResult,
    ActivityInput,
    AnalysisBlock,
    InputMetadata,
    ModalityBundle,
    SpeechAnalysisResult,
)

logger = logging.getLogger(__name__)

_PREVIEW_LEN = 120


class MyUniSentimentPipeline:
    """Routes activities through modality analyzers and returns standardized results.

    Client ``analyze()`` supports text, image, audio, and video with automatic routing.
    CLI/batch ``analyze_activity()`` continues to support text/image/video (+ audio).
    """

    def __init__(
        self,
        text_analyzer: Optional[TextSentimentAnalyzer] = None,
        image_analyzer: Optional[ImageAnalyzer] = None,
        audio_analyzer: Optional[AudioAnalyzer] = None,
        video_analyzer: Optional[VideoAnalyzer] = None,
        fusion_config: FusionConfig = DEFAULT_FUSION,
        video_sampling: VideoSamplingConfig = DEFAULT_VIDEO_SAMPLING,
        video_sampling_strategy: str = "fixed_fps",
        video_debug: bool = False,
    ) -> None:
        self._text_analyzer = text_analyzer or TextSentimentAnalyzer()
        self._image_analyzer = image_analyzer or ImageAnalyzer(
            text_analyzer=self._text_analyzer,
        )
        self._image_analyzer.set_text_analyzer(self._text_analyzer)
        self._audio_analyzer = audio_analyzer or AudioAnalyzer(
            text_analyzer=self._text_analyzer,
        )
        self._audio_analyzer.set_text_analyzer(self._text_analyzer)
        self._fusion_config = fusion_config
        self._video_analyzer = video_analyzer or VideoAnalyzer(
            image_analyzer=self._image_analyzer,
            audio_analyzer=self._audio_analyzer,
            text_analyzer=self._text_analyzer,
            sampling=video_sampling,
            sampling_strategy=video_sampling_strategy,
            fusion_config=fusion_config,
            debug=video_debug,
        )
        self._video_analyzer.set_text_analyzer(self._text_analyzer)

    @property
    def text_analyzer(self) -> TextSentimentAnalyzer:
        return self._text_analyzer

    @property
    def image_analyzer(self) -> ImageAnalyzer:
        return self._image_analyzer

    @property
    def audio_analyzer(self) -> AudioAnalyzer:
        return self._audio_analyzer

    @property
    def video_analyzer(self) -> VideoAnalyzer:
        return self._video_analyzer

    def analyze_speech(self, media_path: object) -> SpeechAnalysisResult:
        """Run the speech branch on an audio/video media path."""
        return self._audio_analyzer.analyze(media_path)  # type: ignore[arg-type]

    def analyze(
        self,
        *,
        text: Optional[str] = None,
        media_path: Optional[str] = None,
        mime_type: Optional[str] = None,
        filename: Optional[str] = None,
        user_id: Optional[str] = None,
        activity_id: Optional[str] = None,
        enabled_modalities: Optional[frozenset[InputType]] = None,
    ) -> RoutedAnalysisResult:
        """Unified client entry: detect input type, then route to real analyzers.

        - TEXT → Twitter-RoBERTa
        - IMAGE → SigLIP 2 zero-shot visual sentiment
        - AUDIO → faster-whisper → RoBERTa transcript sentiment
        - VIDEO → frame SigLIP + speech branch, transparent late fusion

        Never invents scores after failures or empty evidence.
        """
        enabled = CLIENT_ENABLED_MODALITIES if enabled_modalities is None else enabled_modalities
        try:
            detected = InputRouter.detect(
                text=text,
                media_path=media_path,
                mime_type=mime_type,
                filename=filename,
            )
        except ValueError as exc:
            return RoutedAnalysisResult(
                status=CapabilityStatus.VALIDATION_ERROR,
                detected_input=None,
                message=str(exc),
            )

        if not InputRouter.is_enabled(detected.input_type, enabled):
            return RoutedAnalysisResult(
                status=CapabilityStatus.NOT_IMPLEMENTED,
                detected_input=detected.input_type,
                message=InputRouter.not_implemented_message(detected.input_type),
            )

        if detected.input_type == InputType.TEXT:
            assert detected.text is not None
            display_name, model_id = InputRouter.text_model_meta()
            try:
                result = self.analyze_text(
                    detected.text,
                    user_id=user_id,
                    activity_id=activity_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Text analysis failed")
                return RoutedAnalysisResult(
                    status=CapabilityStatus.VALIDATION_ERROR,
                    detected_input=InputType.TEXT,
                    message=f"Text analysis failed: {exc}",
                )
            return RoutedAnalysisResult(
                status=CapabilityStatus.OK,
                detected_input=InputType.TEXT,
                analysis=result,
                model_display_name=display_name,
                model_id=model_id,
            )

        if not detected.media_path:
            return RoutedAnalysisResult(
                status=CapabilityStatus.VALIDATION_ERROR,
                detected_input=detected.input_type,
                message="Upload could not be saved for analysis. Please try again.",
            )

        resolved_id = activity_id or f"ACT-{uuid.uuid4().hex[:8].upper()}"
        try:
            if detected.input_type == InputType.IMAGE:
                result = self._client_analyze_image(
                    detected.media_path,
                    user_id=user_id,
                    activity_id=resolved_id,
                )
                if result.analysis.modalities.visual is None:
                    return RoutedAnalysisResult(
                        status=CapabilityStatus.INSUFFICIENT_EVIDENCE,
                        detected_input=InputType.IMAGE,
                        message=InputRouter.insufficient_evidence_message(InputType.IMAGE),
                    )
                display_name, model_id = InputRouter.visual_model_meta()
                return RoutedAnalysisResult(
                    status=CapabilityStatus.OK,
                    detected_input=InputType.IMAGE,
                    analysis=result,
                    model_display_name=display_name,
                    model_id=model_id,
                )

            if detected.input_type == InputType.AUDIO:
                result = self._client_analyze_audio(
                    detected.media_path,
                    user_id=user_id,
                    activity_id=resolved_id,
                )
                if result.analysis.modalities.speech is None:
                    return RoutedAnalysisResult(
                        status=CapabilityStatus.INSUFFICIENT_EVIDENCE,
                        detected_input=InputType.AUDIO,
                        analysis=result,
                        message=InputRouter.insufficient_evidence_message(InputType.AUDIO),
                        model_display_name=InputRouter.audio_model_meta()[0],
                        model_id=DEFAULT_WHISPER_MODEL,
                    )
                display_name, model_id = InputRouter.audio_model_meta()
                return RoutedAnalysisResult(
                    status=CapabilityStatus.OK,
                    detected_input=InputType.AUDIO,
                    analysis=result,
                    model_display_name=display_name,
                    model_id=f"{model_id} | {DEFAULT_TEXT_MODEL}",
                )

            if detected.input_type == InputType.VIDEO:
                result = self._client_analyze_video(
                    detected.media_path,
                    user_id=user_id,
                    activity_id=resolved_id,
                )
                used = list(result.analysis.fusion.contributing_modalities) if result.analysis.fusion else []
                if not used:
                    return RoutedAnalysisResult(
                        status=CapabilityStatus.INSUFFICIENT_EVIDENCE,
                        detected_input=InputType.VIDEO,
                        analysis=result,
                        message=InputRouter.insufficient_evidence_message(InputType.VIDEO),
                        model_display_name=InputRouter.video_model_meta()[0],
                        model_id=DEFAULT_VISUAL_MODEL,
                    )
                display_name, model_id = InputRouter.video_model_meta()
                return RoutedAnalysisResult(
                    status=CapabilityStatus.OK,
                    detected_input=InputType.VIDEO,
                    analysis=result,
                    model_display_name=display_name,
                    model_id=model_id,
                )
        except (FileNotFoundError, ValueError, FFmpegError, FFmpegNotFoundError) as exc:
            return RoutedAnalysisResult(
                status=CapabilityStatus.VALIDATION_ERROR,
                detected_input=detected.input_type,
                message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s analysis failed", detected.input_type.value)
            return RoutedAnalysisResult(
                status=CapabilityStatus.VALIDATION_ERROR,
                detected_input=detected.input_type,
                message=f"{detected.input_type.value.title()} analysis failed: {exc}",
            )

        return RoutedAnalysisResult(
            status=CapabilityStatus.NOT_IMPLEMENTED,
            detected_input=detected.input_type,
            message=InputRouter.not_implemented_message(detected.input_type),
        )

    def _new_activity(
        self,
        *,
        activity_type: str,
        media_path: str,
        user_id: Optional[str],
        activity_id: str,
        text: Optional[str] = None,
    ) -> ActivityInput:
        return ActivityInput(
            activity_id=activity_id,
            user_id=user_id or "DEMO-USER",
            activity_type=activity_type,  # type: ignore[arg-type]
            text=text,
            media_path=media_path,
            created_at=datetime.now(timezone.utc),
        )

    def _client_analyze_image(
        self,
        media_path: str,
        *,
        user_id: Optional[str],
        activity_id: str,
    ) -> ActivityAnalysisResult:
        activity = self._new_activity(
            activity_type="image",
            media_path=media_path,
            user_id=user_id,
            activity_id=activity_id,
        )
        return self._analyze_image_activity(activity)

    def _client_analyze_audio(
        self,
        media_path: str,
        *,
        user_id: Optional[str],
        activity_id: str,
    ) -> ActivityAnalysisResult:
        speech = self._audio_analyzer.analyze(media_path)
        modalities = ModalityBundle(speech=speech.sentiment)
        if speech.sentiment is not None:
            fusion = fuse_modalities({"speech": speech.sentiment}, config=self._fusion_config)
            overall = fusion.overall
            diagnostics = fusion.diagnostics
        else:
            # Explicit insufficient path: do not invent a neutral sentiment label.
            fusion = fuse_modalities({}, config=self._fusion_config)
            overall = fusion.overall.model_copy(
                update={
                    "details": {
                        **(fusion.overall.details or {}),
                        "insufficient_evidence": True,
                        "reason": "no_speech_or_empty_transcript",
                    },
                },
            )
            diagnostics = fusion.diagnostics.model_copy(
                update={
                    "explanation": (
                        "No usable speech transcript; overall sentiment not derived "
                        "from invented scores."
                    ),
                    "contributing_modalities": [],
                },
            )

        return ActivityAnalysisResult(
            activity_id=activity_id,
            user_id=user_id,
            activity_type="audio",
            input=InputMetadata(media_path=media_path),
            analysis=AnalysisBlock(
                overall=overall,
                modalities=modalities,
                fusion=diagnostics,
                runtime=build_poc_runtime_info(self),
                warnings=list(speech.warnings),
                transcript=speech.transcript,
            ),
        )

    def _client_analyze_video(
        self,
        media_path: str,
        *,
        user_id: Optional[str],
        activity_id: str,
    ) -> ActivityAnalysisResult:
        """Client video path: visual + speech late fusion (equal POC weights)."""
        activity = self._new_activity(
            activity_type="video",
            media_path=media_path,
            user_id=user_id,
            activity_id=activity_id,
        )
        # Reuse frame/ASR analyzers, then fuse visual+speech only for client overall.
        assert activity.media_path is not None
        bundle = self._video_analyzer.analyze(activity.media_path)
        fusion = fuse_modalities(
            {
                "visual": bundle.visual,
                "speech": bundle.speech,
            },
            config=self._fusion_config,
        )
        used = list(fusion.diagnostics.contributing_modalities)
        note = fusion.diagnostics.explanation
        if len(used) == 1:
            note = (
                f"Only {used[0]} evidence was usable; POC overall is derived from "
                f"{used[0]} only. Not a clinically validated fusion formula."
            )
        elif len(used) >= 2:
            note = (
                "Equal-weight POC late fusion over visual + speech "
                "(confidence-weighted). Not client business scoring / not clinically validated."
            )
        diagnostics = fusion.diagnostics.model_copy(update={"explanation": note})

        return ActivityAnalysisResult(
            activity_id=activity_id,
            user_id=user_id,
            activity_type="video",
            input=InputMetadata(media_path=media_path),
            analysis=AnalysisBlock(
                overall=fusion.overall,
                modalities=ModalityBundle(
                    visual=bundle.visual,
                    speech=bundle.speech,
                ),
                fusion=diagnostics,
                runtime=build_poc_runtime_info(self),
                warnings=list(bundle.warnings),
                transcript=bundle.transcript,
                video=bundle.diagnostics,
            ),
        )

    def analyze_text(
        self,
        text: object,
        *,
        user_id: Optional[str] = None,
        activity_id: Optional[str] = None,
    ) -> ActivityAnalysisResult:
        """Analyze a free-form text activity (CLI convenience path)."""
        cleaned = self._text_analyzer.validate_text(text)
        resolved_activity_id = activity_id or f"ACT-{uuid.uuid4().hex[:8].upper()}"
        return self._analyze_text_content(
            cleaned,
            activity_id=resolved_activity_id,
            user_id=user_id,
            created_at=None,
            content_kind=None,
            extra=None,
            media_path=None,
        )

    def analyze_activity(self, activity: ActivityInput) -> ActivityAnalysisResult:
        """Analyze a validated ActivityInput (text, image, or video)."""
        if activity.activity_type == "text":
            assert activity.text is not None
            cleaned = self._text_analyzer.validate_text(activity.text)
            return self._analyze_text_content(
                cleaned,
                activity_id=activity.activity_id,
                user_id=activity.user_id,
                created_at=activity.created_at,
                content_kind=activity.content_kind,
                extra=activity.metadata,
                media_path=activity.media_path,
            )

        if activity.activity_type == "image":
            return self._analyze_image_activity(activity)

        if activity.activity_type == "video":
            return self._analyze_video_activity(activity)

        if activity.activity_type == "audio":
            return self._client_analyze_audio(
                activity.media_path,  # type: ignore[arg-type]
                user_id=activity.user_id,
                activity_id=activity.activity_id,
            )

        raise NotImplementedError(
            f"activity_type={activity.activity_type!r} is not implemented yet",
        )

    def _analyze_text_content(
        self,
        cleaned: str,
        *,
        activity_id: str,
        user_id: Optional[str],
        created_at: object,
        content_kind: object,
        extra: object,
        media_path: Optional[str],
    ) -> ActivityAnalysisResult:
        evidence = self._text_analyzer.analyze(cleaned)
        preview = cleaned if len(cleaned) <= _PREVIEW_LEN else f"{cleaned[:_PREVIEW_LEN]}..."
        fusion = fuse_modalities({"text": evidence}, config=self._fusion_config)

        logger.info(
            "Analyzed text activity_id=%s user_id=%s label=%s score=%.3f",
            activity_id,
            user_id,
            fusion.overall.label,
            fusion.overall.score,
        )

        return ActivityAnalysisResult(
            activity_id=activity_id,
            user_id=user_id,
            activity_type="text",
            input=InputMetadata(
                text_length=len(cleaned),
                text_preview=preview,
                media_path=media_path,
                created_at=created_at,  # type: ignore[arg-type]
                content_kind=content_kind,  # type: ignore[arg-type]
                extra=extra,  # type: ignore[arg-type]
            ),
            analysis=AnalysisBlock(
                overall=fusion.overall,
                modalities=ModalityBundle(text=evidence),
                fusion=fusion.diagnostics,
                runtime=build_poc_runtime_info(self),
            ),
        )

    def _analyze_image_activity(self, activity: ActivityInput) -> ActivityAnalysisResult:
        assert activity.media_path is not None
        media_path = activity.media_path

        caption_evidence = None
        caption_preview = None
        caption_length = None
        if activity.text:
            cleaned_caption = self._text_analyzer.validate_text(activity.text)
            caption_evidence = self._text_analyzer.analyze(cleaned_caption)
            caption_evidence = caption_evidence.model_copy(
                update={
                    "details": {
                        **(caption_evidence.details or {}),
                        "source": "caption",
                    },
                },
            )
            caption_length = len(cleaned_caption)
            caption_preview = (
                cleaned_caption
                if len(cleaned_caption) <= _PREVIEW_LEN
                else f"{cleaned_caption[:_PREVIEW_LEN]}..."
            )

        image_evidence = self._image_analyzer.analyze_path(media_path)

        modalities = ModalityBundle(
            text=caption_evidence,
            visual=image_evidence.visual,
            ocr=image_evidence.ocr_sentiment,
        )
        fusion = fuse_modalities(
            {
                "text": caption_evidence,
                "visual": image_evidence.visual,
                "ocr": image_evidence.ocr_sentiment,
            },
            config=self._fusion_config,
        )

        logger.info(
            "Analyzed image activity_id=%s user_id=%s overall=%s score=%.3f warnings=%s",
            activity.activity_id,
            activity.user_id,
            fusion.overall.label,
            fusion.overall.score,
            len(image_evidence.warnings),
        )

        return ActivityAnalysisResult(
            activity_id=activity.activity_id,
            user_id=activity.user_id,
            activity_type="image",
            input=InputMetadata(
                text_length=caption_length,
                text_preview=caption_preview,
                media_path=media_path,
                created_at=activity.created_at,
                content_kind=activity.content_kind,
                extra=activity.metadata,
            ),
            analysis=AnalysisBlock(
                overall=fusion.overall,
                modalities=modalities,
                fusion=fusion.diagnostics,
                runtime=build_poc_runtime_info(self),
                warnings=list(image_evidence.warnings),
                ocr_text=image_evidence.ocr_text,
            ),
        )

    def _analyze_video_activity(self, activity: ActivityInput) -> ActivityAnalysisResult:
        assert activity.media_path is not None
        media_path = activity.media_path

        caption_evidence = None
        caption_preview = None
        caption_length = None
        if activity.text:
            cleaned_caption = self._text_analyzer.validate_text(activity.text)
            caption_evidence = self._text_analyzer.analyze(cleaned_caption)
            caption_evidence = caption_evidence.model_copy(
                update={
                    "details": {
                        **(caption_evidence.details or {}),
                        "source": "caption",
                    },
                },
            )
            caption_length = len(cleaned_caption)
            caption_preview = (
                cleaned_caption
                if len(cleaned_caption) <= _PREVIEW_LEN
                else f"{cleaned_caption[:_PREVIEW_LEN]}..."
            )

        bundle = self._video_analyzer.analyze(
            media_path,
            caption_sentiment=caption_evidence,
        )

        # Re-fuse so caption is always included even if VideoAnalyzer overall was computed.
        fusion = fuse_modalities(
            {
                "text": caption_evidence,
                "visual": bundle.visual,
                "ocr": bundle.ocr,
                "speech": bundle.speech,
            },
            config=self._fusion_config,
        )

        logger.info(
            "Analyzed video activity_id=%s user_id=%s overall=%s score=%.3f warnings=%s",
            activity.activity_id,
            activity.user_id,
            fusion.overall.label,
            fusion.overall.score,
            len(bundle.warnings),
        )

        return ActivityAnalysisResult(
            activity_id=activity.activity_id,
            user_id=activity.user_id,
            activity_type="video",
            input=InputMetadata(
                text_length=caption_length,
                text_preview=caption_preview,
                media_path=media_path,
                created_at=activity.created_at,
                content_kind=activity.content_kind,
                extra=activity.metadata,
            ),
            analysis=AnalysisBlock(
                overall=fusion.overall,
                modalities=ModalityBundle(
                    text=caption_evidence,
                    visual=bundle.visual,
                    ocr=bundle.ocr,
                    speech=bundle.speech,
                ),
                fusion=fusion.diagnostics,
                runtime=build_poc_runtime_info(self),
                warnings=list(bundle.warnings),
                ocr_text=bundle.ocr_text,
                transcript=bundle.transcript,
                video=bundle.diagnostics,
            ),
        )

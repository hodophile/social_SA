"""
Lightweight speaker diarization for the temporal emotion pipeline.

Strategy (no heavy pyannote model needed):
  1. Take Whisper ASR segments with start/end timestamps
  2. Detect speaker changes via pause gaps (> PAUSE_THRESHOLD_SECONDS)
  3. Optional: use segment embedding similarity for clustering
  4. Label speakers A, B, C...

This is a heuristic approach suitable for CPU/GPU-constrained environments.
For production-grade diarization, replace with pyannote.audio Pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


PAUSE_THRESHOLD_SECONDS = 1.0  # gap > 1s → potential speaker change
MIN_SEGMENT_DURATION = 0.5


@dataclass
class DiarizedSegment:
    """ASR segment with speaker label."""
    start: float
    end: float
    text: str
    speaker: str = "UNKNOWN"
    confidence: Optional[float] = None


@dataclass
class DiarizationResult:
    """Result of diarization over a set of ASR segments."""
    segments: List[DiarizedSegment] = field(default_factory=list)
    num_speakers: int = 0
    speaker_labels: List[str] = field(default_factory=list)


def diarize_segments(
    segments: List[dict],
    pause_threshold: float = PAUSE_THRESHOLD_SECONDS,
) -> DiarizationResult:
    """
    Simple pause-based speaker diarization.

    Args:
        segments: List of {"start": float, "end": float, "text": str}
        pause_threshold: Gap in seconds that triggers a speaker change

    Returns:
        DiarizationResult with speaker labels assigned
    """
    if not segments:
        return DiarizationResult()

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: s["start"])

    diarized: List[DiarizedSegment] = []
    current_speaker_idx = 0
    speaker_names = ["Speaker A"]

    for i, seg in enumerate(sorted_segs):
        # Check gap from previous segment
        if i > 0:
            prev_end = sorted_segs[i - 1]["end"]
            gap = seg["start"] - prev_end
            if gap > pause_threshold:
                # Potential speaker change
                current_speaker_idx += 1
                speaker_name = chr(ord("A") + current_speaker_idx)
                speaker_names.append(f"Speaker {speaker_name}")

        ds = DiarizedSegment(
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            speaker=speaker_names[current_speaker_idx],
        )
        diarized.append(ds)

    # Post-process: merge consecutive segments from same speaker
    merged = _merge_consecutive(diarized)

    unique_speakers = sorted({s.speaker for s in merged})
    return DiarizationResult(
        segments=merged,
        num_speakers=len(unique_speakers),
        speaker_labels=unique_speakers,
    )


def _merge_consecutive(segments: List[DiarizedSegment]) -> List[DiarizedSegment]:
    """Merge consecutive segments from the same speaker."""
    if not segments:
        return []

    merged: List[DiarizedSegment] = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.speaker == last.speaker:
            # Merge
            last.end = seg.end
            last.text = (last.text + " " + seg.text).strip()
        else:
            merged.append(seg)
    return merged


def format_diarization_for_llm(result: DiarizationResult) -> str:
    """Format diarized segments as a narrative string for the LLM prompt."""
    if not result.segments:
        return "No speech detected."

    lines = [f"Detected {result.num_speakers} speaker(s): {', '.join(result.speaker_labels)}"]
    lines.append("")
    for seg in result.segments:
        lines.append(f"[{seg.start:.1f}s-{seg.end:.1f}s] {seg.speaker}: \"{seg.text}\"")
    return "\n".join(lines)


def add_speakers_to_timeline(
    timeline: List,
    diarization: DiarizationResult,
) -> List:
    """
    Enrich timeline ASR events with speaker labels.
    timeline items are dicts with 'time_start', 'time_end', 'type', 'data'.
    """
    if not diarization.segments:
        return timeline

    # Build a lookup: segment start -> speaker
    speaker_by_start = {round(s.start, 2): s.speaker for s in diarization.segments}

    for item in timeline:
        if item.get("type") == "asr" or item.get("event_type") == "asr":
            ts = round(item.get("time_start", item.get("start", 0)), 2)
            # Find closest segment
            speaker = "UNKNOWN"
            for seg in diarization.segments:
                if seg.start <= ts <= seg.end:
                    speaker = seg.speaker
                    break
            # Update data
            if "data" in item:
                item["data"]["speaker"] = speaker
            else:
                item["speaker"] = speaker

    return timeline

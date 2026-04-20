"""Audio analysis utilities for environmental sound detection."""

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-locals,too-many-branches,too-many-statements

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable

import librosa
import numpy as np
from transformers import pipeline as hf_pipeline


@dataclass
class AnalysisPrediction:
    """Structured prediction from the audio analyzer."""

    alert_type: str
    alert_confidence: float
    category: str
    confidence: float
    model_name: str
    detections: list[dict[str, Any]]
    captions: list[dict[str, Any]]
    full_transcript: str


@dataclass(frozen=True)
class SoundDetection:
    """A timestamped sound-event detection."""

    label: str
    category: str
    start_time: float
    end_time: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        """Convert the detection to a JSON-serializable dictionary."""
        return {
            "label": self.label,
            "category": self.category,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class CaptionSegment:
    """A timestamped speech-to-text caption segment."""

    start_time: float
    end_time: float
    text: str
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        """Convert the caption to a JSON-serializable dictionary."""
        return {
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "text": self.text,
            "confidence": round(self.confidence, 4),
        }


def _best_match(
    predictions: list[dict[str, Any]],
    keyword_map: dict[str, tuple[str, ...]],
) -> tuple[str, float]:
    """Pick highest-score category that matches any keyword in label text."""
    best_label = "unknown"
    best_score = 0.0

    for item in predictions:
        raw_label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))

        for canonical_label, keywords in keyword_map.items():
            if any(keyword in raw_label for keyword in keywords) and score > best_score:
                best_label = canonical_label
                best_score = score

    return best_label, best_score


def parse_sound_event_predictions(
    predictions: list[dict[str, Any]] | dict[str, Any],
) -> tuple[str, float, str, float]:
    """Extract normalized alert and category predictions from model outputs."""
    if isinstance(predictions, dict):
        normalized_predictions = [predictions]
    else:
        normalized_predictions = predictions

    alert_keywords = {
        "siren": ("siren", "ambulance", "police", "emergency"),
        "car_horn": ("car horn", "vehicle horn", "horn", "honk", "honking"),
        "bicycle": ("bicycle", "bike", "cycling", "bicyclist"),
        "dog_bark": ("dog bark", "bark", "dog", "bow-wow"),
        "bird_chirp": (
            "bird vocalization",
            "bird call",
            "bird song",
            "chirp",
            "tweet",
            "bird",
        ),
        "traffic_noise": (
            "traffic noise",
            "roadway noise",
            "car passing",
            "vehicle",
            "car",
            "truck",
            "bus",
        ),
        "alarm": ("alarm", "beep", "warning", "smoke detector"),
        "engine": ("engine", "motor", "vehicle"),
        "footsteps": ("footsteps", "steps", "walking"),
        "construction": (
            "jackhammer",
            "drilling",
            "hammer",
            "construction",
            "power tool",
        ),
    }

    category_keywords = {
        "transportation": ("car", "vehicle", "bus", "truck", "bicycle", "bike"),
        "human_activity": ("speech", "talk", "footsteps", "voice", "walking"),
        "animal": ("dog", "cat", "bird", "animal", "chirp", "bark"),
        "safety": ("siren", "alarm", "warning", "emergency"),
        "mechanical": ("engine", "motor", "machinery", "machine", "drill"),
        "ambient": ("traffic", "street", "noise", "outdoor", "urban"),
    }

    alert_type, alert_confidence = _best_match(normalized_predictions, alert_keywords)
    category, category_confidence = _best_match(
        normalized_predictions, category_keywords
    )

    if alert_type == "unknown" and normalized_predictions:
        top_prediction = max(
            normalized_predictions,
            key=lambda item: float(item.get("score", 0.0)),
        )
        top_label = str(top_prediction.get("label", "unknown")).lower()
        alert_type = re.sub(r"[^a-z0-9]+", "_", top_label).strip("_") or "unknown"
        alert_confidence = float(top_prediction.get("score", 0.0))

    if category == "unknown":
        if any(
            token in alert_type
            for token in ("bird", "dog", "bark", "chirp", "cat", "animal")
        ):
            category = "animal"
            category_confidence = max(category_confidence, alert_confidence)
        elif any(
            token in alert_type
            for token in (
                "car",
                "vehicle",
                "truck",
                "bus",
                "traffic",
                "horn",
                "engine",
                "bicycle",
            )
        ):
            category = "transportation"
            category_confidence = max(category_confidence, alert_confidence)

    label_category_map = {
        "music": "ambient",
        "bird_chirp": "animal",
        "dog_bark": "animal",
        "traffic_noise": "transportation",
        "car_horn": "transportation",
        "bicycle": "transportation",
        "engine": "mechanical",
        "construction": "mechanical",
        "siren": "safety",
        "alarm": "safety",
        "footsteps": "human_activity",
    }
    mapped_category = label_category_map.get(alert_type)
    if mapped_category is not None:
        category = mapped_category
        category_confidence = max(category_confidence, alert_confidence)

    confidence = max(alert_confidence, category_confidence)

    return alert_type, alert_confidence, category, confidence


def _contains_speech_prediction(predictions: list[dict[str, Any]]) -> bool:
    """Return True when predictions likely contain speech/human voice."""
    for prediction in predictions:
        label = str(prediction.get("label", "")).lower()
        score = float(prediction.get("score", 0.0))
        if score < 0.2:
            continue
        if any(token in label for token in ("speech", "talk", "voice", "conversation")):
            return True
    return False


def _is_usable_caption_text(text: str) -> bool:
    """Filter obvious ASR hallucinations from non-speech audio."""
    clean_text = text.strip()
    if not clean_text:
        return False

    alnum_chars = sum(character.isalnum() for character in clean_text)
    if alnum_chars / max(len(clean_text), 1) < 0.2:
        return False

    tokens = [token for token in re.split(r"\s+", clean_text.lower()) if token]
    if len(tokens) >= 6 and len(set(tokens)) <= 2:
        return False

    return True


def merge_consecutive_detections(
    detections: list[SoundDetection],
    max_gap_seconds: float = 0.25,
) -> list[SoundDetection]:
    """Merge adjacent detections with the same label into ranges."""
    if not detections:
        return []

    merged: list[SoundDetection] = [detections[0]]
    for detection in detections[1:]:
        previous = merged[-1]
        if (
            detection.label == previous.label
            and detection.category == previous.category
            and detection.start_time <= previous.end_time + max_gap_seconds
        ):
            merged[-1] = SoundDetection(
                label=previous.label,
                category=previous.category,
                start_time=previous.start_time,
                end_time=max(previous.end_time, detection.end_time),
                confidence=max(previous.confidence, detection.confidence),
            )
            continue

        merged.append(detection)

    return merged


def build_caption_segments(
    transcription_result: dict[str, Any],
    chunk_start_seconds: float,
    chunk_duration_seconds: float,
) -> list[CaptionSegment]:
    """Convert a transcription result into caption segments."""
    caption_segments: list[CaptionSegment] = []
    raw_chunks = transcription_result.get("chunks")

    if isinstance(raw_chunks, list) and raw_chunks:
        for chunk in raw_chunks:
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            if not _is_usable_caption_text(text):
                continue

            timestamp = chunk.get("timestamp")
            if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                start_offset = float(timestamp[0] or 0.0)
                end_offset = float(timestamp[1] or chunk_duration_seconds)
            else:
                start_offset = 0.0
                end_offset = chunk_duration_seconds

            caption_segments.append(
                CaptionSegment(
                    start_time=chunk_start_seconds + start_offset,
                    end_time=chunk_start_seconds + end_offset,
                    text=text,
                    confidence=float(
                        chunk.get("score", transcription_result.get("score", 0.0))
                    ),
                )
            )
        return caption_segments

    text = str(transcription_result.get("text", "")).strip()
    if not text:
        return []
    if not _is_usable_caption_text(text):
        return []

    caption_segments.append(
        CaptionSegment(
            start_time=chunk_start_seconds,
            end_time=chunk_start_seconds + chunk_duration_seconds,
            text=text,
            confidence=float(transcription_result.get("score", 0.0)),
        )
    )
    return caption_segments


class HuggingFaceAudioAnalyzer:  # pylint: disable=too-many-instance-attributes
    """Sound-event analyzer backed by a Hugging Face audio-classification model."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        sound_model_id: str,
        asr_model_id: str,
        top_k: int = 20,
        sound_chunk_seconds: float = 10.0,
        sound_hop_seconds: float = 5.0,
        speech_chunk_seconds: float = 5.0,
        enable_captions: bool = False,
        max_audio_seconds: float = 45.0,
        min_alert_confidence: float = 0.03,
    ):
        self.sound_model_id = sound_model_id
        self.asr_model_id = asr_model_id
        self.top_k = top_k
        self.sound_chunk_seconds = sound_chunk_seconds
        self.sound_hop_seconds = sound_hop_seconds
        self.speech_chunk_seconds = speech_chunk_seconds
        self.enable_captions = enable_captions
        self.max_audio_seconds = max_audio_seconds
        self.min_alert_confidence = min_alert_confidence
        self._classifier = None
        self._transcriber = None

    def _ensure_classifier(self):
        if self._classifier is not None:
            return

        self._classifier = hf_pipeline(
            task="audio-classification",
            model=self.sound_model_id,
        )

    def _ensure_transcriber(self):
        if self._transcriber is not None:
            return

        self._transcriber = hf_pipeline(
            task="automatic-speech-recognition",
            model=self.asr_model_id,
        )

    def analyze(
        self,
        audio_path: str,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[
        str, Any
    ]:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        """Run inference for one audio file and return normalized predictions."""

        def report_progress(percent: int, stage: str):
            if progress_callback is None:
                return
            progress_callback(max(0, min(100, percent)), stage)

        file_path = Path(audio_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        report_progress(5, "loading_models")
        self._ensure_classifier()
        if self.enable_captions:
            self._ensure_transcriber()

        report_progress(10, "loading_audio")
        audio, sample_rate = librosa.load(str(file_path), sr=16000, mono=True)
        if self.max_audio_seconds > 0:
            max_samples = int(sample_rate * self.max_audio_seconds)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

        effective_sound_chunk_seconds = max(self.sound_chunk_seconds, 10.0)
        sound_chunk_size = max(1, int(sample_rate * effective_sound_chunk_seconds))
        sound_hop_size = max(1, int(sample_rate * self.sound_hop_seconds))
        speech_chunk_size = max(1, int(sample_rate * self.speech_chunk_seconds))
        audio_duration_seconds = len(audio) / sample_rate
        sound_chunks = list(range(0, len(audio), sound_hop_size))
        total_sound_chunks = max(1, len(sound_chunks))
        detection_progress_start = 10
        detection_progress_end = 80 if self.enable_captions else 95

        raw_detections: list[SoundDetection] = []
        for index, start in enumerate(sound_chunks):
            end = min(len(audio), start + sound_chunk_size)
            chunk = audio[start:end]
            if len(chunk) == 0:
                continue

            if len(chunk) < sound_chunk_size:
                chunk = np.pad(chunk, (0, sound_chunk_size - len(chunk)))

            predictions = self._classifier(
                {"array": chunk, "sampling_rate": sample_rate},
                top_k=self.top_k,
            )
            (
                alert_type,
                alert_confidence,
                category,
                confidence,
            ) = parse_sound_event_predictions(predictions)

            if (
                alert_type == "unknown"
                or alert_confidence <= 0
                or alert_confidence < self.min_alert_confidence
            ):
                continue

            raw_detections.append(
                SoundDetection(
                    label=alert_type,
                    category=category,
                    start_time=start / sample_rate,
                    end_time=min(
                        audio_duration_seconds,
                        (start / sample_rate) + effective_sound_chunk_seconds,
                    ),
                    confidence=confidence,
                )
            )

            completed_ratio = (index + 1) / total_sound_chunks
            progress_value = detection_progress_start + int(
                (detection_progress_end - detection_progress_start) * completed_ratio
            )
            report_progress(progress_value, "detecting_sounds")

        merged_detections = merge_consecutive_detections(raw_detections)

        raw_captions: list[CaptionSegment] = []
        if self.enable_captions:
            speech_chunks = list(range(0, len(audio), speech_chunk_size))
            total_speech_chunks = max(1, len(speech_chunks))
            for index, start in enumerate(speech_chunks):
                end = min(len(audio), start + speech_chunk_size)
                chunk = audio[start:end]
                if len(chunk) == 0:
                    continue

                speech_predictions = self._classifier(
                    {"array": chunk, "sampling_rate": sample_rate},
                    top_k=min(10, self.top_k),
                )
                if isinstance(speech_predictions, dict):
                    speech_predictions = [speech_predictions]

                if not _contains_speech_prediction(speech_predictions):
                    continue

                transcription = self._transcriber(
                    {"array": chunk, "sampling_rate": sample_rate},
                )
                raw_captions.extend(
                    build_caption_segments(
                        transcription_result=transcription,
                        chunk_start_seconds=index * self.speech_chunk_seconds,
                        chunk_duration_seconds=min(
                            audio_duration_seconds,
                            self.speech_chunk_seconds,
                        ),
                    )
                )

                caption_ratio = (index + 1) / total_speech_chunks
                report_progress(80 + int(15 * caption_ratio), "transcribing")

        captions = [caption.as_dict() for caption in raw_captions]
        full_transcript = " ".join(
            caption.text.strip() for caption in raw_captions if caption.text.strip()
        ).strip()

        if merged_detections:
            primary = max(merged_detections, key=lambda item: item.confidence)
            alert_type = primary.label
            alert_confidence = primary.confidence
            category = primary.category
            confidence = primary.confidence
        else:
            alert_type = "unknown"
            alert_confidence = 0.0
            category = "unknown"
            confidence = 0.0

        parsed_prediction = AnalysisPrediction(
            alert_type=alert_type,
            alert_confidence=alert_confidence,
            category=category,
            confidence=confidence,
            model_name=self.sound_model_id,
            detections=[detection.as_dict() for detection in merged_detections],
            captions=captions,
            full_transcript=full_transcript,
        )
        report_progress(100, "completed")
        return parsed_prediction.__dict__

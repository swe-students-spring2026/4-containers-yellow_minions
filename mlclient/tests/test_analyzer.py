"""Tests for analyzer parsing helpers."""

from analyzer import (
    CaptionSegment,
    SoundDetection,
    build_caption_segments,
    merge_consecutive_detections,
    parse_sound_event_predictions,
)


def test_parse_sound_event_predictions():
    predictions = [
        {"label": "car horn", "score": 0.91},
        {"label": "vehicle", "score": 0.84},
    ]

    alert_type, alert_confidence, category, confidence = parse_sound_event_predictions(
        predictions
    )

    assert alert_type == "car_horn"
    assert category == "transportation"
    assert alert_confidence == 0.91
    assert confidence == 0.91


def test_parse_predictions_defaults_to_unknown():
    predictions = [{"label": "ambient_noise", "score": 0.99}]

    alert_type, alert_confidence, category, confidence = parse_sound_event_predictions(
        predictions
    )

    assert alert_type == "unknown"
    assert category == "unknown"
    assert alert_confidence == 0.0
    assert confidence == 0.0


def test_merge_consecutive_detections_groups_adjacent_alerts():
    detections = [
        SoundDetection(
            label="dog_bark",
            category="animal",
            start_time=0.0,
            end_time=1.0,
            confidence=0.7,
        ),
        SoundDetection(
            label="dog_bark",
            category="animal",
            start_time=1.0,
            end_time=2.0,
            confidence=0.8,
        ),
        SoundDetection(
            label="car_horn",
            category="transportation",
            start_time=4.0,
            end_time=5.0,
            confidence=0.9,
        ),
    ]

    merged = merge_consecutive_detections(detections)

    assert len(merged) == 2
    assert merged[0].label == "dog_bark"
    assert merged[0].start_time == 0.0
    assert merged[0].end_time == 2.0
    assert merged[0].confidence == 0.8
    assert merged[1].label == "car_horn"


def test_build_caption_segments_uses_chunk_timestamps():
    transcription = {
        "text": "dog barking",
        "chunks": [
            {
                "timestamp": (0.1, 1.4),
                "text": "dog barking",
                "score": 0.91,
            }
        ],
    }

    segments = build_caption_segments(transcription, 12.0, 5.0)

    assert len(segments) == 1
    assert segments[0] == CaptionSegment(
        start_time=12.1,
        end_time=13.4,
        text="dog barking",
        confidence=0.91,
    )


def test_build_caption_segments_falls_back_to_chunk_text():
    transcription = {"text": "car honking", "score": 0.82}

    segments = build_caption_segments(transcription, 4.0, 5.0)

    assert len(segments) == 1
    assert segments[0].text == "car honking"
    assert segments[0].start_time == 4.0
    assert segments[0].end_time == 9.0

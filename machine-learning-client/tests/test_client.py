"""Tests for ML job worker behavior."""

from pathlib import Path

import mongomock

from client import MLClient


class StubAnalyzer:
    """Simple deterministic analyzer stub."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    def analyze(self, audio_path: str):
        if self.should_fail:
            raise RuntimeError("model failed")

        return {
            "alert_type": "car_horn",
            "alert_confidence": 0.81,
            "category": "transportation",
            "confidence": 0.88,
            "model_name": "stub-model",
            "captions": [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "text": "car honking",
                    "confidence": 0.88,
                }
            ],
            "full_transcript": "car honking",
            "audio_path_seen": audio_path,
        }


def test_run_once_processes_pending_job(tmp_path: Path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"fake-wav-bytes")

    mongo_client = mongomock.MongoClient()
    database = mongo_client["sound_alerts"]
    job_id = database.analysis_jobs.insert_one(
        {
            "audio_path": str(audio_file),
            "status": "pending",
            "created_at": 1,
        }
    ).inserted_id

    worker = MLClient(
        mongo_uri="mongodb://unused",
        db_name="sound_alerts",
        analyzer=StubAnalyzer(),
        mongo_client=mongo_client,
    )

    result = worker.run_once()

    assert result is not None
    prediction = database.predictions.find_one({"job_id": job_id})
    assert prediction is not None
    assert prediction["alert_type"] == "car_horn"
    assert prediction["category"] == "transportation"
    assert prediction["full_transcript"] == "car honking"
    assert prediction["captions"]

    updated_job = database.analysis_jobs.find_one({"_id": job_id})
    assert updated_job["status"] == "done"
    assert "prediction_id" in updated_job


def test_run_once_marks_failed_when_analyzer_errors(tmp_path: Path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"fake-wav-bytes")

    mongo_client = mongomock.MongoClient()
    database = mongo_client["sound_alerts"]
    job_id = database.analysis_jobs.insert_one(
        {
            "audio_path": str(audio_file),
            "status": "pending",
            "created_at": 1,
        }
    ).inserted_id

    worker = MLClient(
        mongo_uri="mongodb://unused",
        db_name="sound_alerts",
        analyzer=StubAnalyzer(should_fail=True),
        mongo_client=mongo_client,
    )

    try:
        worker.run_once()
    except RuntimeError:
        pass

    updated_job = database.analysis_jobs.find_one({"_id": job_id})
    assert updated_job["status"] == "failed"
    assert "error" in updated_job
    assert database.predictions.count_documents({}) == 0


def test_run_once_returns_none_when_no_jobs():
    mongo_client = mongomock.MongoClient()

    worker = MLClient(
        mongo_uri="mongodb://unused",
        db_name="sound_alerts",
        analyzer=StubAnalyzer(),
        mongo_client=mongo_client,
    )

    result = worker.run_once()
    assert result is None

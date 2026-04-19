"""Mongo-backed ML worker that classifies environmental sounds."""

from __future__ import annotations

import argparse
import logging
import os
import time
import tempfile
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient, ReturnDocument
from gridfs import GridFSBucket

from analyzer import HuggingFaceAudioAnalyzer


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class MLClient:
    """Background worker for sound-analysis jobs in MongoDB."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mongo_uri: str,
        db_name: str,
        analyzer: Any,
        jobs_collection_name: str = "analysis_jobs",
        predictions_collection_name: str = "predictions",
        mongo_client: MongoClient | None = None,
    ):
        self.mongo_client = mongo_client or MongoClient(mongo_uri)
        self.database = self.mongo_client[db_name]
        self.jobs = self.database[jobs_collection_name]
        self.predictions = self.database[predictions_collection_name]
        self.analyzer = analyzer
        self.bucket = GridFSBucket(self.database, bucket_name="audio_files")

    def _download_audio(self, job):
        """Download audio from GridFS to a temp file."""
        gridfs_id = job.get("gridfs_file_id")
        if gridfs_id is None:
            raise ValueError("Job is missing gridfs_file_id")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            self.bucket.download_to_stream(gridfs_id, tmp)
            return tmp.name

    def fetch_next_job(self) -> dict[str, Any] | None:
        """Atomically claim one pending job."""
        return self.jobs.find_one_and_update(
            {"status": "pending"},
            {
                "$set": {
                    "status": "processing",
                    "started_at": utc_now(),
                }
            },
            sort=[("created_at", 1)],
            return_document=ReturnDocument.AFTER,
        )

    def process_job(self, job: dict[str, Any]) -> dict[str, Any]:
        """Process one job and update collections with outputs."""
        job_id = job["_id"]
        audio_path = self._download_audio(job)

        try:
            prediction_output = self.analyzer.analyze(audio_path)
            prediction_doc = {
                "job_id": job_id,
                "audio_path": audio_path,
                "created_at": utc_now(),
                **prediction_output,
            }
            inserted_prediction = self.predictions.insert_one(prediction_doc)

            self.jobs.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "status": "done",
                        "completed_at": utc_now(),
                        "prediction_id": inserted_prediction.inserted_id,
                    }
                },
            )
            logging.info("Processed job %s successfully", job_id)
            return prediction_doc
        except Exception as exc:  # pylint: disable=broad-except
            self.jobs.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "status": "failed",
                        "completed_at": utc_now(),
                        "error": str(exc),
                    }
                },
            )
            logging.exception("Failed processing job %s: %s", job_id, exc)
            raise

    def run_once(self) -> dict[str, Any] | None:
        """Process a single pending job if available."""
        job = self.fetch_next_job()
        if job is None:
            logging.debug("No pending jobs found")
            return None
        return self.process_job(job)

    def run_forever(self, poll_interval_seconds: int = 5):
        """Continuously poll for jobs and process them."""
        logging.info("ML worker started. Poll interval: %ss", poll_interval_seconds)
        while True:
            try:
                self.run_once()
            except Exception:  # pylint: disable=broad-except
                pass
            time.sleep(poll_interval_seconds)


def build_client_from_env() -> tuple[MLClient, int]:
    """Create MLClient from environment variables."""
    load_dotenv()

    mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    db_name = os.getenv("MONGO_DB_NAME", "sound_alerts")
    sound_model_id = os.getenv(
        "HF_SOUND_MODEL_ID",
        "MIT/ast-finetuned-audioset-10-10-0.4593",
    )
    asr_model_id = os.getenv("HF_ASR_MODEL_ID", "openai/whisper-tiny.en")
    poll_interval_seconds = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))

    analyzer = HuggingFaceAudioAnalyzer(
        sound_model_id=sound_model_id,
        asr_model_id=asr_model_id,
    )
    client = MLClient(
        mongo_uri=mongo_uri,
        db_name=db_name,
        analyzer=analyzer,
    )
    return client, poll_interval_seconds


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run sound-alert analysis worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one pending job and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    client, poll_interval_seconds = build_client_from_env()

    if args.once:
        client.run_once()
    else:
        client.run_forever(poll_interval_seconds=poll_interval_seconds)


if __name__ == "__main__":
    main()

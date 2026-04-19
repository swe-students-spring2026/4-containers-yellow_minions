"""Tests for the Flask web app routes."""

import io
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import pytest
import mongomock
from bson import ObjectId


@pytest.fixture
def mock_gridfs():
    """Patch GridFSBucket so create_app doesn't need real GridFS."""
    mock_bucket = MagicMock()
    mock_bucket.upload_from_stream.return_value = ObjectId()
    with patch("app.GridFSBucket", return_value=mock_bucket):
        yield mock_bucket


@pytest.fixture
def mongo_client():
    """Provide a mongomock client."""
    return mongomock.MongoClient()


@pytest.fixture
def client(mongo_client, mock_gridfs):
    """Create a test Flask client with mocked DB."""
    from app import create_app

    test_app = create_app(mongo_client=mongo_client, db_name="test_db")
    test_app.config["TESTING"] = True
    with test_app.test_client() as test_client:
        yield test_client


@pytest.fixture
def db(mongo_client):
    """Direct access to the test database."""
    return mongo_client["test_db"]


class TestIndexRoute:
    def test_index_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_index_contains_upload_form(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert "Upload" in html
        assert "form" in html.lower()

    def test_index_accepts_audio_files(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert "audio" in html or "media" in html


class TestUploadRoute:
    def test_upload_rejects_missing_file(self, client):
        response = client.post("/upload")
        assert response.status_code == 400

    def test_upload_rejects_empty_filename(self, client):
        data = {"media": (io.BytesIO(b""), "")}
        response = client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        assert response.status_code == 400

    def test_upload_returns_error_json_on_missing_file(self, client):
        response = client.post("/upload")
        json_data = response.get_json()
        assert json_data["success"] is False
        assert "missing file" in json_data["error"]

    def test_upload_redirects_on_success(self, client, mock_gridfs):
        file_id = ObjectId()
        mock_gridfs.upload_from_stream.return_value = file_id
        data = {"media": (io.BytesIO(b"fake audio data"), "test.mp3")}
        response = client.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        assert response.status_code == 302

    def test_upload_creates_job_in_db(self, client, db, mock_gridfs):
        file_id = ObjectId()
        mock_gridfs.upload_from_stream.return_value = file_id
        data = {"media": (io.BytesIO(b"fake audio data"), "test.mp3")}
        client.post("/upload", data=data, content_type="multipart/form-data")
        job = db["analysis_jobs"].find_one()
        assert job is not None
        assert job["status"] == "pending"
        assert job["original_filename"] == "test.mp3"

    def test_upload_stores_gridfs_file_id(self, client, db, mock_gridfs):
        file_id = ObjectId()
        mock_gridfs.upload_from_stream.return_value = file_id
        data = {"media": (io.BytesIO(b"fake audio data"), "sample.wav")}
        client.post("/upload", data=data, content_type="multipart/form-data")
        job = db["analysis_jobs"].find_one()
        assert job["gridfs_file_id"] == file_id


class TestAnalysisRoute:
    def test_analysis_returns_404_for_missing_job(self, client):
        fake_id = str(ObjectId())
        response = client.get(f"/analysis/{fake_id}")
        assert response.status_code == 404

    def test_analysis_shows_pending_status(self, client, db):
        job_id = db["analysis_jobs"].insert_one(
            {
                "status": "pending",
                "original_filename": "clip.mp3",
                "gridfs_file_id": ObjectId(),
                "media_type": "audio/mpeg",
                "created_at": datetime.now(timezone.utc),
            }
        ).inserted_id
        response = client.get(f"/analysis/{job_id}")
        assert response.status_code == 200
        html = response.data.decode()
        assert "pending" in html
        assert "clip.mp3" in html

    def test_analysis_shows_prediction_when_done(self, client, db):
        job_id = db["analysis_jobs"].insert_one(
            {
                "status": "done",
                "original_filename": "siren.mp3",
                "gridfs_file_id": ObjectId(),
                "media_type": "audio/mpeg",
                "created_at": datetime.now(timezone.utc),
            }
        ).inserted_id
        db["predictions"].insert_one(
            {
                "job_id": job_id,
                "alert_type": "siren",
                "alert_confidence": 0.95,
                "category": "safety",
                "confidence": 0.95,
                "model_name": "test-model",
                "detections": [
                    {
                        "label": "siren",
                        "category": "safety",
                        "start_time": 0.0,
                        "end_time": 2.0,
                        "confidence": 0.95,
                    }
                ],
                "captions": [
                    {
                        "start_time": 0.0,
                        "end_time": 2.0,
                        "text": "ambulance siren",
                        "confidence": 0.88,
                    }
                ],
                "full_transcript": "ambulance siren",
            }
        )
        response = client.get(f"/analysis/{job_id}")
        html = response.data.decode()
        assert response.status_code == 200
        assert "siren" in html
        assert "95%" in html
        assert "safety" in html
        assert "ambulance siren" in html

    def test_analysis_shows_failed_message(self, client, db):
        job_id = db["analysis_jobs"].insert_one(
            {
                "status": "failed",
                "original_filename": "bad.mp3",
                "gridfs_file_id": ObjectId(),
                "media_type": "audio/mpeg",
                "created_at": datetime.now(timezone.utc),
            }
        ).inserted_id
        response = client.get(f"/analysis/{job_id}")
        html = response.data.decode()
        assert response.status_code == 200
        assert "failed" in html.lower()

    def test_analysis_auto_refreshes_when_pending(self, client, db):
        job_id = db["analysis_jobs"].insert_one(
            {
                "status": "pending",
                "original_filename": "wait.mp3",
                "gridfs_file_id": ObjectId(),
                "media_type": "audio/mpeg",
                "created_at": datetime.now(timezone.utc),
            }
        ).inserted_id
        response = client.get(f"/analysis/{job_id}")
        html = response.data.decode()
        assert 'http-equiv="refresh"' in html


class TestHistoryRoute:
    def test_history_returns_200(self, client):
        response = client.get("/history")
        assert response.status_code == 200

    def test_history_shows_empty_message(self, client):
        response = client.get("/history")
        html = response.data.decode()
        assert "No analyses yet" in html

    def test_history_lists_past_jobs(self, client, db):
        db["analysis_jobs"].insert_one(
            {
                "status": "done",
                "original_filename": "horn.mp3",
                "gridfs_file_id": ObjectId(),
                "media_type": "audio/mpeg",
                "created_at": datetime.now(timezone.utc),
            }
        )
        response = client.get("/history")
        html = response.data.decode()
        assert "horn.mp3" in html
        assert "done" in html

    def test_history_shows_prediction_data(self, client, db):
        job_id = db["analysis_jobs"].insert_one(
            {
                "status": "done",
                "original_filename": "bark.mp3",
                "gridfs_file_id": ObjectId(),
                "media_type": "audio/mpeg",
                "created_at": datetime.now(timezone.utc),
            }
        ).inserted_id
        db["predictions"].insert_one(
            {
                "job_id": job_id,
                "alert_type": "dog_bark",
                "alert_confidence": 0.82,
                "category": "animal",
                "confidence": 0.82,
                "model_name": "test-model",
                "detections": [],
                "captions": [],
                "full_transcript": "",
            }
        )
        response = client.get("/history")
        html = response.data.decode()
        assert "dog_bark" in html
        assert "82%" in html
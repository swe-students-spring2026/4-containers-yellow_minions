"""Flask web app for sound-alert uploads and results."""

import os
from datetime import datetime, timezone
from uuid import uuid4

from flask import Flask, jsonify, render_template, request
from gridfs import GridFSBucket
from pymongo import MongoClient
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sound_alerts")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]
bucket = GridFSBucket(db, bucket_name="audio_files")
analysis_jobs_collection = db["analysis_jobs"]


@app.route("/")
def index():
    """Homepage."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload media file and create an analysis job."""
    uploaded_file = request.files.get("media")

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"success": False, "error": "missing file"}), 400

    filename = secure_filename(uploaded_file.filename)
    unique_filename = f"{uuid4()}_{filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    uploaded_file.save(file_path)

    with open(file_path, "rb") as media_stream:
        gridfs_file_id = bucket.upload_from_stream(
            filename=filename,
            source=media_stream,
            metadata={"content_type": uploaded_file.content_type},
        )

    job_document = {
        "audio_path": file_path,
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
        "media_path": file_path,
        "media_type": uploaded_file.content_type,
        "original_filename": filename,
        "duration_seconds": None,
        "gridfs_file_id": gridfs_file_id,
    }
    inserted_job = analysis_jobs_collection.insert_one(job_document)

    return jsonify(
        {
            "success": True,
            "filename": filename,
            "job_id": str(inserted_job.inserted_id),
            "gridfs_file_id": str(gridfs_file_id),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

"""Flask web app for sound-alert uploads and results."""

import os
from uuid import uuid4
from datetime import datetime, timezone
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from pymongo import MongoClient

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sound_alerts")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]

analysis_jobs_collection = db["analysis_jobs"]


@app.route("/")
def index():
    """Homepage."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload media file."""
    uploaded_file = request.files.get("media")

    if uploaded_file is None or uploaded_file.filename == "":
        return "No file uploaded.", 400

    filename = secure_filename(uploaded_file.filename)
    unique_filename = f"{uuid4()}_{filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    uploaded_file.save(file_path)
    job_document = {
        "audio_path": file_path,
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
        "media_path": file_path,
        "original_filename": filename,
        "duration_seconds": None,
    }
    analysis_jobs_collection.insert_one(job_document)

    return f"Saved file to: {file_path}"


if __name__ == "__main__":
    app.run(debug=True)

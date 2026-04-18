"""Flask web app for sound-alert uploads and results."""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# from uuid import uuid4

from flask import Flask, jsonify, render_template, request, redirect, url_for, send_file
from gridfs import GridFSBucket
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId
from werkzeug.utils import secure_filename
from mlclient.analyzer import HuggingFaceAudioAnalyzer
from datetime import datetime, timezone
from dotenv import load_dotenv

sys.stdout.reconfigure(line_buffering=True)
load_dotenv()
app = Flask(__name__)

"""Please use .env file for db connection."""
# MONGO_URI = mongodb+srv://{username}:{password}@cluster.m91q1zi.mongodb.net/?appName=Cluster
# MONGO_DB_NAME = audio_description

# Change these ids to .env
analyzer = HuggingFaceAudioAnalyzer(
    sound_model_id=os.getenv("HF_SOUND_MODEL_ID"),
    asr_model_id=os.getenv("HF_ASR_MODEL_ID")
)

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client[os.getenv("MONGO_DB_NAME")]
bucket = GridFSBucket(db, bucket_name="audio_files")
analysis_jobs_collection = db["analysis_jobs"]
analysis_results_collection = db["analysis_results"]


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

    try:
        gridfs_file_id = bucket.upload_from_stream(
            filename=filename,
            source=uploaded_file.stream,
            metadata={"content_type": uploaded_file.content_type},
        )

        job_document = {
            "status": "pending",
            "created_at": datetime.now(timezone.utc),
            "media_type": uploaded_file.content_type,
            "original_filename": filename,
            "duration_seconds": None,
            "gridfs_file_id": gridfs_file_id,
        }
        
        inserted_job = analysis_jobs_collection.insert_one(job_document)
        return redirect(url_for("analysis_page", job_id=str(inserted_job.inserted_id)))

    except PyMongoError as e:
        print(f"database error: {e}")
        return jsonify({"success": False, "error": "database error"}), 500

    except IOError as e:
        print(f"io error: {e}")
        return jsonify({"success": False, "error": "file io error"}), 500


@app.route("/analysis/<job_id>")
def analysis_page(job_id):
    """Render the analysis page for a given job ID with audio playback and analysis under."""
    audio = analysis_jobs_collection.find_one({"_id": ObjectId(job_id)})
    
    results = None
    if audio["status"] == "done":
            results = analysis_results_collection.find_one({"job_id": ObjectId(job_id)})

    return render_template(
        "analysis.html",
        filename=audio["original_filename"],
        gridfs_id=str(audio["gridfs_file_id"]),
        content_type=audio["media_type"],
        job_id=job_id,
        results=results
    )


@app.route("/playback/<gridfs_id>", methods=["GET"])
def playback(gridfs_id):
    """Route with no static file, just a url to play an audio file from gridfs."""
    file = bucket.open_download_stream(ObjectId(gridfs_id))
    print(file)
    return send_file(
        file,
        mimetype=file.metadata.get("content_type", "application/octet-stream"),
        as_attachment=False,
        download_name=file.filename,
    )



if __name__ == "__main__":
    app.run(debug=True)

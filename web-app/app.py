"""Flask web app for sound-alert uploads and results."""

import os

# import sys
from datetime import datetime, timezone

# from uuid import uuid4

from flask import Flask, jsonify, render_template, request, redirect, url_for, send_file
from gridfs import GridFSBucket
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# sys.stdout.reconfigure(line_buffering=True)
load_dotenv()
app = Flask(__name__)

"""Please use .env file for db connection."""
# MONGO_URI = mongodb+srv://{username}:{password}@cluster.m91q1zi.mongodb.net/?appName=Cluster
# MONGO_DB_NAME = audio_description

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client[os.getenv("MONGO_DB_NAME")]
bucket = GridFSBucket(db, bucket_name="audio_files")
analysis_jobs_collection = db["analysis_jobs"]
predictions_collection = db["predictions"]


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
    # unique_filename = f"{uuid4()}_{filename}"

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
    job = analysis_jobs_collection.find_one({"_id": ObjectId(job_id)})
    if job is None:
        return jsonify({"success": False, "error": "job not found"}), 404

    prediction = predictions_collection.find_one({"job_id": ObjectId(job_id)})

    return render_template(
        "analysis.html",
        filename=job.get("original_filename", "unknown"),
        gridfs_id=str(job.get("gridfs_file_id", "")),
        content_type=job.get("media_type", "audio/mpeg"),
        status=job.get("status", "pending"),
        prediction=prediction,
    )


@app.route("/history")
def history():
    """Show all past analysis jobs with their results."""
    all_jobs = list(analysis_jobs_collection.find().sort("created_at", -1).limit(50))
    for job in all_jobs:
        job["_id_str"] = str(job["_id"])
        pred = predictions_collection.find_one({"job_id": job["_id"]})
        job["prediction"] = pred
    return render_template("history.html", jobs=all_jobs)


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
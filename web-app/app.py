"""Flask web app for sound-alert uploads and results."""

import os
from datetime import datetime, timezone

# from uuid import uuid4

from flask import Flask, jsonify, render_template, request
from gridfs import GridFSBucket
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

""" please use .env file for db connection """
# MONGO_URI = mongodb+srv://{username}:{password}@cluster.m91q1zi.mongodb.net/?appName=Cluster
# MONGO_DB_NAME = audio_description

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client[os.getenv("MONGO_DB_NAME")]
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

        return jsonify(
            {
                "success": True,
                "filename": filename,
                "job_id": str(inserted_job.inserted_id),
                "gridfs_file_id": str(gridfs_file_id),
            }
        )

    except PyMongoError as e:
        print(f"database error: {e}")
        return jsonify({"success": False, "error": "database error"}), 500

    except IOError as e:
        print(f"io error: {e}")
        return jsonify({"success": False, "error": "file io error"}), 500


if __name__ == "__main__":
    app.run(debug=True)

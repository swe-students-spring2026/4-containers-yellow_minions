"""Flask web app for sound-alert uploads and results."""

from collections import Counter
import os

# import sys
from datetime import datetime, timezone

# from uuid import uuid4

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
)
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


def _counter_rows(counter: Counter, total: int, limit: int = 10):
    rows = []
    for label, count in counter.most_common(limit):
        percent = round((count / total) * 100, 1) if total > 0 else 0.0
        rows.append({"label": label, "count": count, "percent": percent})
    return rows


def _build_stats(predictions):  # pylint: disable=too-many-locals
    sound_counter = Counter()
    category_counter = Counter()
    alert_confidence_sum = 0.0
    alert_confidence_count = 0
    detection_confidence_sum = 0.0
    detection_confidence_count = 0
    analyzed_duration_seconds = 0.0

    for prediction in predictions:
        alert_confidence_sum += float(prediction.get("alert_confidence", 0.0) or 0.0)
        alert_confidence_count += 1

        detections = prediction.get("detections") or []
        if detections:
            max_end = max(
                float(item.get("end_time", 0.0) or 0.0) for item in detections
            )
            analyzed_duration_seconds += max_end

        for item in detections:
            label = str(item.get("label", "unknown") or "unknown")
            category = str(item.get("category", "unknown") or "unknown")
            confidence = float(item.get("confidence", 0.0) or 0.0)

            sound_counter[label] += 1
            category_counter[category] += 1
            detection_confidence_sum += confidence
            detection_confidence_count += 1

    total_detections = sum(sound_counter.values())
    return {
        "total_predictions": len(predictions),
        "total_detections": total_detections,
        "analyzed_duration_seconds": round(analyzed_duration_seconds, 2),
        "avg_alert_confidence": (
            alert_confidence_sum / alert_confidence_count
            if alert_confidence_count > 0
            else 0.0
        ),
        "avg_detection_confidence": (
            detection_confidence_sum / detection_confidence_count
            if detection_confidence_count > 0
            else 0.0
        ),
        "top_sounds": _counter_rows(sound_counter, total_detections),
        "top_categories": _counter_rows(
            category_counter,
            sum(category_counter.values()),
            limit=8,
        ),
    }


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
            "progress_percent": 0,
            "progress_stage": "queued",
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
        job_id=job_id,
        filename=job.get("original_filename", "unknown"),
        gridfs_id=str(job.get("gridfs_file_id", "")),
        content_type=job.get("media_type", "audio/mpeg"),
        status=job.get("status", "pending"),
        progress_percent=int(job.get("progress_percent", 0)),
        progress_stage=job.get("progress_stage", "queued"),
        is_video=job.get("media_type", "").startswith("video/"),
        prediction=prediction,
    )


@app.route("/dashboard")
def dashboard():
    """Show aggregate analytics across successful clips and optionally one clip."""
    done_jobs = list(
        analysis_jobs_collection.find({"status": "done"}).sort("created_at", -1)
    )
    failed_jobs_count = analysis_jobs_collection.count_documents({"status": "failed"})

    done_job_ids = [job["_id"] for job in done_jobs]
    if done_job_ids:
        all_predictions = list(
            predictions_collection.find({"job_id": {"$in": done_job_ids}})
        )
    else:
        all_predictions = []

    global_stats = _build_stats(all_predictions)

    selected_job = None
    selected_prediction = None
    selected_job_id = request.args.get("job_id")
    if selected_job_id:
        try:
            selected_object_id = ObjectId(selected_job_id)
            selected_job = analysis_jobs_collection.find_one(
                {"_id": selected_object_id}
            )
        except Exception:  # pylint: disable=broad-except
            selected_job = None

    if selected_job is None and done_jobs:
        selected_job = done_jobs[0]

    clip_stats = None
    if selected_job is not None:
        selected_prediction = predictions_collection.find_one(
            {"job_id": selected_job["_id"]}
        )
        if selected_prediction is not None:
            clip_stats = _build_stats([selected_prediction])

    return render_template(
        "dashboard.html",
        selected_job=selected_job,
        selected_prediction=selected_prediction,
        clip_stats=clip_stats,
        global_stats=global_stats,
        done_jobs_count=len(done_jobs),
        failed_jobs_count=failed_jobs_count,
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
    content_type = file.metadata.get("content_type", "application/octet-stream")
    range_header = request.headers.get("Range")

    if range_header:
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0]) if byte_range[0] else 0
        end = (
            int(byte_range[1])
            if len(byte_range) > 1 and byte_range[1]
            else file.length - 1
        )
        end = min(end, file.length - 1)
        chunk_size = end - start + 1

        file.seek(start)
        data = file.read(chunk_size)
        response = Response(data, 206, mimetype=content_type, direct_passthrough=True)
        response.headers.add("Content-Range", f"bytes {start}-{end}/{file.length}")
        response.headers.add("Accept-Ranges", "bytes")
        response.headers.add("Content-Length", str(chunk_size))
        return response

    return send_file(
        file,
        mimetype=content_type,
        as_attachment=False,
        download_name=file.filename,
    )


if __name__ == "__main__":
    app.run(debug=True)

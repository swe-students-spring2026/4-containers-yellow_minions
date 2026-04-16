"""Flask web app for sound-alert uploads and results."""

import os
import sys
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from gridfs import GridFSBucket
from dotenv import load_dotenv

sys.stdout.reconfigure(line_buffering=True)
load_dotenv()

# mongodb connection
# use .env file to connect to atlas
# MONGO_DB_NAME - name of db collection

client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB_NAME")]
bucket = GridFSBucket(db, bucket_name="audio_files")

app = Flask(__name__)


@app.route("/")
def index():
    """Homepage"""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Send Audio recording to be analyzed"""
    file = request.files["audiofile"]

    if not file or file.filename == "":
        return jsonify({"success": False, "error": "missing file"}), 400
    print(file)

    # https://www.mongodb.com/docs/languages/python/pymongo-driver/current/crud/gridfs/
    with bucket.open_upload_stream(
        filename=file.filename, metadata={"content_type": file.content_type}
    ) as grid_in:
        file.save(grid_in)

    return jsonify({"success": True, "filename": file.filename})

    # except Exception as err:
    #     print("err:", err)
    #     return jsonify({"success": False, "error": str(err)}), 400


if __name__ == "__main__":
    app.run(debug=True)

"""Flask web app for sound-alert uploads and results."""

from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
load_dotenv()

uri = os.getenv("MONGO_URI")
print(uri)
client = MongoClient(uri)

app = Flask(__name__)

@app.route("/")
def index():
    """Homepage"""
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """Send Audio recording to be analyzed"""
    try:
        print(request.files)
        file = request.files["audiofile"]
        print(file)
        print(file.content_type)

        return jsonify({"success": True})

    except Exception as err:
        print("err:", err)
        return jsonify({"success": False, "error": str(err)}), 400

if __name__ == "__main__":
    app.run(debug=True)

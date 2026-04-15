"""Flask web app for audio upload."""

from flask import Flask, render_template
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    """Homepage."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="localhost", port=3000, debug=True)

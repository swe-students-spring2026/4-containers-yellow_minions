"""Flask web app for sound-alert uploads and results."""

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    """Homepage."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

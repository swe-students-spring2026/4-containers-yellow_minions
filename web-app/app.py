"""Flask web app for audio upload."""

from flask import Flask, render_template


app = Flask(__name__)


@app.route("/")
def index():
    """Homepage."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
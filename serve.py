"""Production server for the Tempest FWI Predictor.

    python serve.py        -> http://localhost:5000

Uses Waitress, a production WSGI server that runs on Windows, macOS and Linux
(gunicorn doesn't run on Windows). The Docker image uses gunicorn instead.
"""
import os

from waitress import serve

from app.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Tempest FWI Predictor running on http://localhost:{port} (Waitress)")
    serve(app, host="0.0.0.0", port=port, threads=4)

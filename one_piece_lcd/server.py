"""Simple Flask server for visualizing face detection results on video."""

import json
from pathlib import Path

from flask import Flask, Response, send_file, send_from_directory

from .constants.paths import VIDEOS_DIR

# Get the project root (one level up from one_piece_lcd package)
PROJECT_ROOT = Path(__file__).parent.parent

# Get the static directory path
STATIC_DIR = Path(__file__).parent / "static"

# Resolve VIDEOS_DIR to absolute path from project root
VIDEOS_DIR_ABS = PROJECT_ROOT / VIDEOS_DIR

app = Flask(__name__, static_folder=str(STATIC_DIR))

# Video FPS (hardcoded for now, could be extracted from video metadata)
VIDEO_FPS = 24.0


def get_video_dir(video_id: str) -> Path:
    """Get the directory for a video by ID."""
    return VIDEOS_DIR_ABS / video_id


def get_latest_detections_file(video_dir: Path) -> Path | None:
    """Get the most recent face-detections JSON file in a video directory."""
    detection_files = sorted(
        video_dir.glob("face-detections-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return detection_files[0] if detection_files else None


@app.route("/static/<path:filename>")
def serve_static(filename: str) -> Response:
    """Serve static files (JS, CSS, etc.)."""
    return send_from_directory(STATIC_DIR, filename)


@app.route("/video/<video_id>")
def video_page(video_id: str) -> str | tuple[str, int]:
    """Serve the video player page with canvas overlay."""
    video_dir = get_video_dir(video_id)
    
    if not video_dir.exists():
        return f"Video '{video_id}' not found", 404
    
    # Read and render the HTML template
    template_path = STATIC_DIR / "viewer.html"
    with open(template_path) as f:
        html = f.read()
    
    # Simple template substitution
    html = html.replace("{{ video_id }}", video_id)
    html = html.replace("{{ video_fps }}", str(VIDEO_FPS))
    
    return html


@app.route("/api/video/<video_id>/file")
def video_file(video_id: str) -> Response:
    """Serve the video file."""
    video_dir = get_video_dir(video_id)
    mp4_path = video_dir / f"{video_id}.mp4"
    mkv_path = video_dir / f"{video_id}.mkv"

    if mp4_path.exists():
        return send_file(mp4_path, mimetype="video/mp4")
    elif mkv_path.exists():
        # mkv file may not play in all browsers, but serve it anyway
        return send_file(mkv_path, mimetype="video/x-matroska")
    else:
        return Response("Video file not found", status=404)


@app.route("/api/video/<video_id>/detections")
def video_detections(video_id: str) -> Response:
    """Serve the latest face detections JSON."""
    video_dir = get_video_dir(video_id)
    
    if not video_dir.exists():
        return Response("Video directory not found", status=404)
    
    detections_file = get_latest_detections_file(video_dir)
    
    if not detections_file:
        return Response("No detections file found", status=404)
    
    with open(detections_file) as f:
        data = json.load(f)
    
    return Response(
        json.dumps(data),
        mimetype="application/json"
    )


def run_server(host: str = "0.0.0.0", port: int = 3000, debug: bool = True) -> None:
    """Run the Flask development server."""
    print(f"\n🏴‍☠️ Face Detection Viewer running at http://localhost:{port}/video/sample\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()

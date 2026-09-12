# YOLO Studio

An interactive object detection and tracking workspace built with **HTML, CSS and vanilla JavaScript**, backed by a local **Flask + YOLOv8** API. Streamlit is no longer part of the active application.

## Run on Windows

Double-click `start.bat` and open http://127.0.0.1:8501.

For a fresh installation (Python 3.12 recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

The current environment reuses previously installed scientific packages, with additional web dependencies installed locally. A fresh environment can use the requirements above. The first model load downloads official Ultralytics weights if they are missing.

On this machine the downloaded project has a very long folder path, exceeding Windows native-library path limits. `runtime-python.txt` points to a short directory junction for the same `.venv`; `start.bat` automatically uses it. No packages in other projects were changed. For commands in this checkout, use `& (Get-Content runtime-python.txt) ...` in place of `.\.venv\Scripts\python.exe`. On a fresh machine, keep the checkout path short or configure a short virtual environment. The local runtime file is ignored by Git.

Sign in with the existing `users.json` account. The original demo credentials are `admin` / `admin123` unless changed. Account → User management supports adding/removing users and changing the administrator password. Sessions reset when the server restarts.

## Features

- Drag-and-drop image upload and a bundled office sample.
- Actual YOLOv8 detection and instance segmentation, with 80 COCO classes.
- Confidence threshold and searchable multi-class selection.
- Select an inspector row or bounding box to highlight an object.
- Toggle boxes, labels and segmentation masks independently.
- Video upload, frame-by-frame analysis, optional ByteTrack tracking, progress and cancellation.
- Browser webcam access, live frame inference, and explicit stop controls. Camera access stops when the tab goes into the background.
- Annotated PNG / MP4 downloads, JSON and CSV exports, and browser Print / Save as PDF.
- Responsive dark/light interface, keyboard focus, reduced-motion support, and Ctrl+Enter to run.

## Processing limits and behavior

- The server binds to `127.0.0.1`; this is a local application.
- Maximum upload: 100 MB. Images: up to 30 megapixels, resized to at most 1920 pixels on the longest edge for inference.
- Videos: up to 2 minutes, resized to at most 960 pixels. One video job runs at a time. Output is H.264 MP4 **without audio**.
- Video totals are detections summed across frames, not distinct object counts. Tracker IDs are reported separately and may change after occlusion.
- Live webcam inference is sequential; speed depends on this computer. It does not promise a fixed frame rate or track identities across webcam frames.
- Images are processed in memory. Video uploads are deleted after completion/cancellation; rendered videos are stored in `outputs/web`. Inactive job records and outputs older than one hour are removed when a new video is submitted; leftovers from an earlier server session may remain on disk.
- Model predictions are probabilistic. This preserves the local demo's SHA-256 credential format; it is not production authentication. The server has same-origin mutation checks and authenticated, owner-scoped video access.

## Files

- `static/index.html`: interface structure.
- `static/styles.css`: responsive visual theme and print styles.
- `static/app.js`: uploads, canvas overlays, camera, jobs, exports and account UI.
- `server.py`: Flask endpoints, inference, authentication and video pipeline.
- `app.py`, `run_app.py`, `start.bat`: launchers.
- `weights/`: detection and segmentation models.
- `legacy-streamlit/`: preserved originals for rollback; not imported or run by the new app.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest tests -q
node --check static/app.js
```

Integration checks use the actual weights, sample image, tracker and video encoder. No mock detections.

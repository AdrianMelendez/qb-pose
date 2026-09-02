# qb-pose

Quarterback throwing-motion biomechanics analysis using [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).

It detects a 33-point body skeleton on an image or video of a QB throwing and computes:

- **Elbow angle** at the throwing arm, drawn live on every frame
- **Wrist speed** — throwing-hand speed over time (a proxy for release speed, not ball speed)
- **Release time(s)** — detected as the peak(s) in wrist speed; a clip can contain several throwing reps
- **Hip-shoulder separation** — rotational lead of the hips over the shoulders, a proxy for throwing efficiency

The reusable analysis code lives in [`analysis.py`](analysis.py); `demo.ipynb` is a thin driver notebook on top of it.

## Example

Static skeleton overlay on a still frame:

![Annotated pose skeleton](examples/annotated-image.jpg)

Wrist speed, hip-shoulder separation, and a release flash burned into the video:

![Metrics overlay tracked through a throw, with a RELEASE flash at the detected release point](examples/annotated-throw.gif)

Wrist speed and hip-shoulder separation over the full clip (three throwing reps), aligned to the detected release:

![Wrist speed and hip-shoulder separation chart](examples/throw-metrics-chart.png)

## Usage

1. Install dependencies: `uv sync`
2. Open `demo.ipynb` (e.g. `uv run jupyter lab`) and run all cells.

Each section saves its annotated output to disk (`output_image.jpg`, `qb-throw-annotated.mp4`, `throw-metrics.png`) and additionally opens a live preview window if a display is available.

## Multi-view (side & front)

`analyze_video()` and `compare_views()` in `analysis.py` work on a video from any camera angle — MediaPipe assigns landmarks by the subject's own anatomical left/right, not by where the camera is standing. Drop a `qb-throw-side.mp4` and/or `qb-throw-front.mp4` of the same throwing session next to `qb-throw.mp4` and re-run the "Multi-View Comparison" cell in the notebook to overlay all views (each aligned to its own detected release).

This isn't true synchronized 3D triangulation — that needs calibrated, timestamp-aligned cameras filming the same instant — but it does let you sanity-check whether independent single-view measurements roughly agree, since a 2D-plane joint angle or speed estimate can be distorted by which way the camera happens to be facing.

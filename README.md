# qb-pose

Quarterback throwing-motion biomechanics analysis using [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).

It detects a 33-point body skeleton on an image or video of a QB throwing, draws the throwing-arm triangle (shoulder → elbow → wrist), and computes the elbow angle at each frame so you can see how it opens up through the release.

## Example

Static skeleton overlay on a still frame:

![Annotated pose skeleton](examples/annotated-image.jpg)

Elbow angle tracked live across the throwing motion:

![Elbow angle tracked through a throw](examples/annotated-throw.gif)

## Usage

The analysis lives in `demo.ipynb`:

1. Install dependencies: `uv sync`
2. Open `demo.ipynb` (e.g. `uv run jupyter lab`) and run all cells.

Each section saves its annotated output to disk (`output_image.jpg`, `qb-throw-annotated.mp4`) and additionally opens a live preview window if a display is available.

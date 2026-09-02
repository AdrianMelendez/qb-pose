"""Streamlit front-end for qb-pose: upload a throw, see the metrics.

Run with: uv run streamlit run app.py
"""

import tempfile
from pathlib import Path

import cv2
import streamlit as st

from analysis import (
    MODEL_PATH_DEFAULT,
    analyze_video,
    compare_views,
    render_annotated_video,
    summarize_consistency,
    summarize_releases,
    track_ball_release,
)

MAX_DURATION_S = 60  # soft cap so one upload can't tie up the app for minutes

st.set_page_config(page_title="qb-pose", page_icon="🏈", layout="centered")
st.title("🏈 QB Throwing Motion Analysis")
st.write(
    "Upload a video of a quarterback throwing. This detects the throwing-arm "
    "skeleton and reports elbow angle, wrist speed, release time(s), hip-shoulder "
    "separation, trunk lean, and stride length, plus consistency across reps and "
    "(experimentally) real ball speed - optionally from up to three camera angles "
    "of the same throwing session."
)


def _probe_duration_s(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_total / fps if fps else 0.0


@st.cache_data(show_spinner=False)
def run_analysis(video_bytes: bytes, view: str):
    """Analyze one uploaded clip and render its annotated video.

    Cached on the raw upload bytes + view label, so re-running the app (any
    widget interaction re-runs the whole script) doesn't reprocess an
    unchanged upload.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name
    output_path = video_path + ".annotated.mp4"

    try:
        duration_s = _probe_duration_s(video_path)
        if duration_s > MAX_DURATION_S:
            raise ValueError(
                f"Clip is {duration_s:.0f}s, longer than the {MAX_DURATION_S}s limit for this demo."
            )

        analysis = analyze_video(video_path, model_path=MODEL_PATH_DEFAULT, view=view)
        events = summarize_releases(analysis)
        consistency = summarize_consistency(events)
        ball_releases = track_ball_release(video_path, analysis, model_path=MODEL_PATH_DEFAULT)
        render_annotated_video(video_path, output_path, analysis, model_path=MODEL_PATH_DEFAULT)
        annotated_bytes = Path(output_path).read_bytes()
        return analysis, events, consistency, ball_releases, annotated_bytes
    finally:
        Path(video_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


back_file = st.file_uploader("Back view (required)", type=["mp4", "mov", "m4v"])
col1, col2 = st.columns(2)
with col1:
    side_file = st.file_uploader("Side view (optional)", type=["mp4", "mov", "m4v"])
with col2:
    front_file = st.file_uploader("Front view (optional)", type=["mp4", "mov", "m4v"])

if back_file is None:
    st.info("Upload at least a back-view clip to get started.")
    st.stop()

uploads = {"back": back_file, "side": side_file, "front": front_file}
results = {}

for view, uploaded in uploads.items():
    if uploaded is None:
        continue
    with st.spinner(f"Analyzing {view} view... this can take up to a minute for longer clips."):
        try:
            results[view] = run_analysis(uploaded.getvalue(), view)
        except ValueError as e:
            st.error(f"{view.capitalize()} view: {e}")

if not results:
    st.stop()

CONSISTENCY_LABELS = {
    "speed_mps": "Release speed (m/s)",
    "peak_separation_deg": "Peak hip-shoulder separation (deg)",
    "elbow_angle_release_deg": "Elbow angle @release (deg)",
    "trunk_lean_release_deg": "Trunk lean @release (deg)",
    "stride_length_m": "Stride length (m)",
}

for view, (analysis, events, consistency, ball_releases, annotated_bytes) in results.items():
    st.subheader(f"{view.capitalize()} view")
    st.video(annotated_bytes)

    if events:
        st.table(
            [
                {
                    "Time (s)": f"{e['time_s']:.2f}",
                    "Release speed": f"{e['speed_mps']:.2f} m/s ({e['speed_mph']:.1f} mph)",
                    "Peak hip-shoulder sep.": f"{e['peak_separation_deg']:.1f} deg",
                    "Elbow @release": f"{e['elbow_angle_release_deg']:.1f} deg",
                    "Elbow @max cocking": f"{e['elbow_angle_max_cocking_deg']:.1f} deg",
                    "Trunk lean": f"{e['trunk_lean_release_deg']:.1f} deg",
                    "Stride length": f"{e['stride_length_m']:.2f} m",
                }
                for e in events
            ]
        )

        if consistency:
            with st.expander("Consistency across reps (mean ± std, CV%)"):
                st.table(
                    [
                        {
                            "Metric": CONSISTENCY_LABELS.get(key, key),
                            "Mean ± std": f"{stat['mean']:.2f} ± {stat['std']:.2f}",
                            "CV%": f"{stat['cv_pct']:.0f}%" if stat["cv_pct"] is not None else "n/a",
                        }
                        for key, stat in consistency.items()
                    ]
                )

        with st.expander("Ball tracking (experimental)"):
            st.caption(
                "Best-effort tracking of the football itself, not the throwing hand - gives a "
                "real speed/angle instead of the wrist-speed proxy above, but only when it can "
                "find the ball with confidence. Works best on an uncluttered background and a "
                "side-view angle where the ball moves laterally across the frame; a back view "
                "often finds nothing, since the ball then moves mostly in depth."
            )
            if ball_releases:
                st.table(
                    [
                        {
                            "Frame": b.frame_idx,
                            "Ball speed": f"{b.speed_mps:.2f} m/s ({b.speed_mph:.1f} mph)",
                            "Launch angle": f"{b.angle_deg:.1f} deg",
                            "Detections": len(b.positions),
                        }
                        for b in ball_releases
                    ]
                )
            else:
                st.write("No confident ball track found in this clip.")
    else:
        st.warning("No clear release event detected in this clip.")

st.subheader("Wrist speed & hip-shoulder separation")
fig = compare_views({view: r[0] for view, r in results.items()})
st.pyplot(fig)

st.caption(
    "Release speed is throwing-hand speed, a proxy for ball speed, not the ball itself. "
    "When multiple views are uploaded, each is aligned to its own detected release rather "
    "than synchronized/triangulated - see the README for why."
)

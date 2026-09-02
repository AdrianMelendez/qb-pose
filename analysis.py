"""Reusable pose/biomechanics analysis for qb-pose.

Kept out of the notebook so the same logic can be driven from anywhere else
(a CLI, a batch job, a future web backend) instead of being copy-pasted.
"""

import os
import subprocess
import sys
from dataclasses import dataclass

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np

MODEL_PATH_DEFAULT = "./pose_landmarker.task"

# Throwing-arm/torso landmark indices. MediaPipe assigns these by the
# subject's own anatomical left/right (not by camera position), so the same
# indices are correct regardless of which side of the subject the camera is on.
R_SHOULDER, L_SHOULDER = 12, 11
R_ELBOW = 14
R_WRIST = 16
R_HIP, L_HIP = 24, 23


def has_display():
    """Return True if a GUI window can actually be shown (X11/Wayland/Windows/macOS)."""
    if os.name == "nt" or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def smooth(values, window=5):
    """Simple moving average to reduce per-frame landmark jitter."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def transverse_angle(p_from, p_to):
    """Rotation angle (degrees) of the line p_from->p_to in the horizontal (x-z) plane."""
    return np.degrees(np.arctan2(p_to.z - p_from.z, p_to.x - p_from.x))


def find_peaks(values, min_distance, min_height):
    """Local maxima at least min_distance samples apart and above min_height."""
    peaks = []
    for i in range(len(values)):
        window = values[max(0, i - min_distance) : i + min_distance + 1]
        if values[i] == window.max() and values[i] >= min_height:
            if not peaks or i - peaks[-1] >= min_distance:
                peaks.append(i)
    return peaks


def put_outlined_text(canvas, text, org, scale=0.6, color=(255, 255, 255), thickness=2):
    """cv2.putText with a black outline, for legibility against any background."""
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def visualize_qb_manual(image_path, detection_result):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    if not detection_result.pose_landmarks:
        return image

    # MediaPipe Pose Connections (Symmetry for QB: Shoulder to Elbow, Elbow to Wrist)
    # These are the indices for the right arm: 12 (Shoulder), 14 (Elbow), 16 (Wrist)
    CONNECTIONS = [(12, 14), (14, 16), (11, 13), (13, 15), (12, 11), (24, 23)]

    for pose_landmarks in detection_result.pose_landmarks:
        for lm in pose_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

        for start_idx, end_idx in CONNECTIONS:
            start_lm = pose_landmarks[start_idx]
            end_lm = pose_landmarks[end_idx]
            start_point = (int(start_lm.x * w), int(start_lm.y * h))
            end_point = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(image, start_point, end_point, (255, 0, 0), 2)

    return image


def visualize_qb_analysis(frame, detection_result):
    """
    frame: The BGR numpy array from cv2.VideoCapture
    detection_result: The result object from detector.detect_for_video
    """
    canvas = frame.copy()
    h, w, _ = canvas.shape

    if not detection_result or not detection_result.pose_landmarks:
        return canvas

    for pose_landmarks in detection_result.pose_landmarks:
        def get_pix(idx):
            lm = pose_landmarks[idx]
            return (int(lm.x * w), int(lm.y * h))

        p_shoulder = get_pix(R_SHOULDER)
        p_elbow = get_pix(R_ELBOW)
        p_wrist = get_pix(R_WRIST)

        cv2.line(canvas, p_shoulder, p_elbow, (255, 255, 255), 2)
        cv2.line(canvas, p_elbow, p_wrist, (255, 255, 255), 2)

        for pt in [p_shoulder, p_elbow, p_wrist]:
            cv2.circle(canvas, pt, 6, (0, 255, 0), -1)

        ba = np.array(p_shoulder) - np.array(p_elbow)
        bc = np.array(p_wrist) - np.array(p_elbow)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

        label = f"{angle:.1f} deg"
        text_origin = (p_elbow[0] + 15, p_elbow[1])
        cv2.putText(canvas, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return canvas


def draw_metrics_hud(canvas, wrist_speed_mps=None, separation_deg=None, is_release=False):
    """Overlay wrist speed / hip-shoulder separation / a release flash in the corner."""
    y = 30
    if wrist_speed_mps is not None:
        put_outlined_text(canvas, f"Wrist speed: {wrist_speed_mps:.2f} m/s ({wrist_speed_mps * 2.237:.1f} mph)", (10, y))
        y += 28
    if separation_deg is not None:
        put_outlined_text(canvas, f"Hip-shoulder separation: {separation_deg:.1f} deg", (10, y))
        y += 28
    if is_release:
        put_outlined_text(canvas, "RELEASE", (10, y + 12), scale=1.0, color=(0, 0, 255), thickness=3)
    return canvas


def transcode_to_h264(path):
    """Re-encode a video in place to H.264/yuv420p.

    cv2.VideoWriter's 'mp4v' (MPEG-4 Part 2) output plays in players like VLC
    but not in browsers, which is what any web front-end needs. Uses the
    static ffmpeg binary bundled by imageio-ffmpeg so this doesn't depend on
    ffmpeg being separately installed on the host.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_path = path + ".h264.mp4"
    subprocess.run(
        [
            ffmpeg_exe, "-y", "-i", path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-loglevel", "error", tmp_path,
        ],
        check=True,
    )
    os.replace(tmp_path, path)


@dataclass
class ThrowAnalysis:
    view: str
    fps: float
    frame_width: int
    frame_height: int
    timestamps_s: np.ndarray
    separation_deg: np.ndarray
    speed_timestamps: np.ndarray
    wrist_speed: np.ndarray
    release_frames: list
    release_video_frames: set
    frame_indices: np.ndarray


def _make_landmarker_options(model_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    return PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
    )


def analyze_video(video_path, model_path=MODEL_PATH_DEFAULT, view="back"):
    """Run pose detection over a video and compute wrist speed, release
    events, and hip-shoulder separation from the throwing arm/torso.

    Uses MediaPipe's metric-scale pose_world_landmarks (not the normalized
    image-space ones) so the result isn't distorted by camera perspective.
    """
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    options = _make_landmarker_options(model_path)

    frame_indices, timestamps_s = [], []
    wrist_xyz, shoulder_angles, hip_angles = [], [], []

    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open this file as a video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_count / fps) * 1000)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_world_landmarks:
                world = result.pose_world_landmarks[0]
                frame_indices.append(frame_count)
                timestamps_s.append(timestamp_ms / 1000)
                wrist_xyz.append((world[R_WRIST].x, world[R_WRIST].y, world[R_WRIST].z))
                shoulder_angles.append(transverse_angle(world[L_SHOULDER], world[R_SHOULDER]))
                hip_angles.append(transverse_angle(world[L_HIP], world[R_HIP]))

            frame_count += 1

        cap.release()

    if len(timestamps_s) < 2:
        raise ValueError(
            "No pose could be detected in enough of this video to analyze it - "
            "make sure a person is clearly visible for most of the clip."
        )

    frame_indices = np.array(frame_indices)
    timestamps_s = np.array(timestamps_s)
    wrist_xyz = np.array(wrist_xyz).reshape(-1, 3)
    shoulder_angles = smooth(shoulder_angles)
    hip_angles = smooth(hip_angles)

    # Hip-shoulder separation: rotational difference between the shoulder and
    # hip lines in the transverse plane - a proxy for kinematic-sequence
    # efficiency (bigger/faster separation generally means a more efficient throw).
    separation_deg = np.abs(((shoulder_angles - hip_angles) + 180) % 360 - 180)

    wrist_x, wrist_y, wrist_z = (smooth(wrist_xyz[:, i]) for i in range(3))
    dt = np.diff(timestamps_s)
    wrist_speed = smooth(np.sqrt(np.diff(wrist_x) ** 2 + np.diff(wrist_y) ** 2 + np.diff(wrist_z) ** 2) / dt, window=3)
    speed_timestamps = timestamps_s[1:]
    speed_frame_indices = frame_indices[1:]

    # Release time(s): wrist speed peaks right around release, then
    # decelerates through the follow-through. A clip may contain several
    # throwing reps, so this finds every such peak rather than assuming one throw.
    min_distance_frames = max(1, int(round(0.6 * fps)))
    speed_threshold = wrist_speed.mean() + wrist_speed.std()
    release_frames = find_peaks(wrist_speed, min_distance_frames, speed_threshold)
    release_video_frames = {int(speed_frame_indices[i]) for i in release_frames}

    return ThrowAnalysis(
        view=view,
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
        timestamps_s=timestamps_s,
        separation_deg=separation_deg,
        speed_timestamps=speed_timestamps,
        wrist_speed=wrist_speed,
        release_frames=release_frames,
        release_video_frames=release_video_frames,
        frame_indices=frame_indices,
    )


def summarize_releases(analysis: ThrowAnalysis):
    """Per-release-event dicts: time, throwing-hand speed, peak separation."""
    events = []
    for idx in analysis.release_frames:
        t = analysis.speed_timestamps[idx]
        speed_mps = analysis.wrist_speed[idx]

        window_mask = (analysis.timestamps_s >= t - 1.0) & (analysis.timestamps_s <= t)
        sep_window = analysis.separation_deg[window_mask]
        sep_ts_window = analysis.timestamps_s[window_mask]
        peak_sep_i = int(np.argmax(sep_window))

        events.append(
            {
                "time_s": t,
                "speed_mps": speed_mps,
                "speed_mph": speed_mps * 2.237,
                "peak_separation_deg": sep_window[peak_sep_i],
                "separation_lead_ms": (t - sep_ts_window[peak_sep_i]) * 1000,
            }
        )
    return events


def render_annotated_video(
    video_path, output_path, analysis: ThrowAnalysis, model_path=MODEL_PATH_DEFAULT, flash_frames=6, web_safe=True
):
    """Write an annotated video: skeleton + elbow angle + wrist speed +
    hip-shoulder separation + a brief RELEASE flash at each detected event.

    Also opens a live preview window if a display is available. When
    web_safe (the default), the output is transcoded to H.264 afterwards so
    it plays inline in browsers - cv2.VideoWriter's own 'mp4v' output often won't.
    """
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    options = _make_landmarker_options(model_path)

    sep_by_frame = dict(zip(analysis.frame_indices.tolist(), analysis.separation_deg.tolist()))
    speed_by_frame = dict(zip(analysis.frame_indices[1:].tolist(), analysis.wrist_speed.tolist()))
    flash = {f for rf in analysis.release_video_frames for f in range(rf, rf + flash_frames)}

    show_live = has_display()

    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (analysis.frame_width, analysis.frame_height)
        )

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_count / fps) * 1000)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            annotated = visualize_qb_analysis(frame, result)
            draw_metrics_hud(
                annotated,
                wrist_speed_mps=speed_by_frame.get(frame_count),
                separation_deg=sep_by_frame.get(frame_count),
                is_release=frame_count in flash,
            )
            writer.write(annotated)

            if show_live:
                cv2.imshow("QB Biomechanics Feed", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1

        cap.release()
        writer.release()
        if show_live:
            cv2.destroyAllWindows()

    if web_safe:
        transcode_to_h264(output_path)

    return frame_count


def compare_views(results: "dict[str, ThrowAnalysis]"):
    """Overlay wrist speed and hip-shoulder separation curves from one or
    more named views (e.g. {"back": ..., "side": ..., "front": ...}).

    Each view is aligned so its own first detected release lands at t=0.
    True synchronized 3D triangulation isn't possible without calibrated,
    timestamp-aligned cameras - this instead lets you sanity-check whether
    independent single-view measurements roughly agree, since a 2D joint
    angle/speed estimate can be distorted by which way the camera is facing.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = plt.cm.tab10.colors

    for i, (view_name, analysis) in enumerate(results.items()):
        color = colors[i % len(colors)]
        anchor = analysis.speed_timestamps[analysis.release_frames[0]] if analysis.release_frames else 0.0
        ax1.plot(analysis.speed_timestamps - anchor, analysis.wrist_speed, color=color, label=view_name)
        ax2.plot(analysis.timestamps_s - anchor, analysis.separation_deg, color=color, label=view_name)

    ax1.set_ylabel("Wrist speed (m/s)")
    ax1.axvline(0, color="black", linestyle=":", alpha=0.5)
    ax1.legend()

    ax2.set_ylabel("Hip-shoulder separation (deg)")
    ax2.set_xlabel("Time relative to each view's first detected release (s)")
    ax2.axvline(0, color="black", linestyle=":", alpha=0.5)

    fig.suptitle("Wrist speed & hip-shoulder separation" + (" (multi-view)" if len(results) > 1 else ""))
    fig.tight_layout()
    return fig

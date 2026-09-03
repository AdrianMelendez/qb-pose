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

# Downscale wider frames to this before running pose detection or writing
# output - MediaPipe resizes internally anyway, so accuracy is largely
# unaffected, but reading/inferring/encoding a 4K or 1080p phone video at
# full resolution is dramatically slower (and was the likely cause of a
# processing run that looked "stuck" rather than just slow).
MAX_FRAME_WIDTH = 960

# Throwing-arm/torso landmark indices. MediaPipe assigns these by the
# subject's own anatomical left/right (not by camera position), so the same
# indices are correct regardless of which side of the subject the camera is on.
R_SHOULDER, L_SHOULDER = 12, 11
R_ELBOW = 14
R_WRIST = 16
R_HIP, L_HIP = 24, 23
R_ANKLE, L_ANKLE = 28, 27

# Real-world long axis of an NFL football (~11in), used only to size the
# search for a ball-like blob - not for any speed/distance calculation.
BALL_LONG_AXIS_M = 0.28


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


def angle_between(v1, v2):
    """Angle (degrees) between two 3D vectors."""
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _as_vec(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])


def _target_size(width, height, max_width=MAX_FRAME_WIDTH):
    """Downscaled (width, height) preserving aspect ratio, or the size
    unchanged if it's already at or under max_width."""
    if width <= max_width or width <= 0:
        return width, height
    scale = max_width / width
    return max_width, max(1, int(round(height * scale)))


def _read_resized(cap, target_width, target_height):
    """cap.read(), resized to (target_width, target_height) if that differs
    from the frame's actual size."""
    ret, frame = cap.read()
    if not ret:
        return ret, frame
    h, w = frame.shape[:2]
    if (w, h) != (target_width, target_height):
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return ret, frame


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
    elbow_angle_deg: np.ndarray
    trunk_lean_deg: np.ndarray
    foot_separation_m: np.ndarray


def _make_landmarker_options(model_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    return PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
    )


def analyze_video(video_path, model_path=MODEL_PATH_DEFAULT, view="back", on_progress=None):
    """Run pose detection over a video and compute wrist speed, release
    events, and hip-shoulder separation from the throwing arm/torso.

    Uses MediaPipe's metric-scale pose_world_landmarks (not the normalized
    image-space ones) so the result isn't distorted by camera perspective.

    on_progress(frame_count, total_frames), if given, is called after every
    frame - total_frames may be None if the container doesn't report a
    reliable frame count. Useful for a caller (e.g. a web UI) to show real
    progress on a long clip instead of a silent wait.
    """
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    options = _make_landmarker_options(model_path)

    frame_indices, timestamps_s = [], []
    wrist_xyz, shoulder_angles, hip_angles = [], [], []
    elbow_angle_deg, trunk_lean_deg, foot_separation_m = [], [], []

    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open this file as a video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width, frame_height = _target_size(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        frame_count = 0
        while cap.isOpened():
            ret, frame = _read_resized(cap, frame_width, frame_height)
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

                # Elbow (flexion) angle, in 3D world space rather than the 2D
                # pixel angle drawn on the video, so it isn't view-dependent.
                shoulder_v = _as_vec(world[R_SHOULDER]) - _as_vec(world[R_ELBOW])
                wrist_v = _as_vec(world[R_WRIST]) - _as_vec(world[R_ELBOW])
                elbow_angle_deg.append(angle_between(shoulder_v, wrist_v))

                # Trunk lean: angle of the hip-to-shoulder line from vertical.
                mid_hip = (_as_vec(world[L_HIP]) + _as_vec(world[R_HIP])) / 2
                mid_shoulder = (_as_vec(world[L_SHOULDER]) + _as_vec(world[R_SHOULDER])) / 2
                trunk_lean_deg.append(angle_between(mid_shoulder - mid_hip, np.array([0.0, -1.0, 0.0])))

                # Foot separation: distance between the ankles, a proxy for
                # stride length that peaks around front-foot plant.
                foot_separation_m.append(float(np.linalg.norm(_as_vec(world[L_ANKLE]) - _as_vec(world[R_ANKLE]))))

            frame_count += 1
            if on_progress:
                on_progress(frame_count, total_frames)

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
    elbow_angle_deg = smooth(elbow_angle_deg)
    trunk_lean_deg = smooth(trunk_lean_deg)
    foot_separation_m = smooth(foot_separation_m)

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
        elbow_angle_deg=elbow_angle_deg,
        trunk_lean_deg=trunk_lean_deg,
        foot_separation_m=foot_separation_m,
    )


def _find_movement_onset(wrist_speed, release_idx, fps, idle_multiplier=1.5, min_idle_s=0.15, max_lookback_s=3.0):
    """Index into wrist_speed where the throwing motion began: the end of
    the last sustained "idle" stretch before this release, within a bounded
    lookback window.

    "Idle" means wrist speed close to the *clip's own* resting baseline (the
    whole-clip 10th percentile), not a threshold relative to this throw's
    own peak speed. Scaling by peak speed sounds adaptive, but a throwing
    motion has its own brief pause mid-windup (the "max cocking" pause,
    still well above true rest) - for a hard throw, even a small fraction
    of its peak speed sits above that pause, so a peak-relative threshold
    ends up finding the cocking pause instead of the true start of the
    motion (empirically, ~100ms "time to release" instead of a plausible
    ~1-2s, on the bundled sample clip).
    """
    max_lookback_frames = int(round(max_lookback_s * fps))
    lo = max(0, release_idx - max_lookback_frames)
    window = wrist_speed[lo:release_idx]
    if len(window) == 0:
        return lo

    baseline = float(np.percentile(wrist_speed, 10))
    threshold = baseline * idle_multiplier
    min_idle_frames = max(1, int(round(min_idle_s * fps)))

    below = window <= threshold
    onset_offset = 0  # fallback: no qualifying idle stretch in the window - motion was already underway
    i = 0
    while i < len(below):
        if below[i]:
            j = i
            while j < len(below) and below[j]:
                j += 1
            if (j - i) >= min_idle_frames:
                onset_offset = j  # keep the *last* qualifying idle->moving transition found
            i = j
        else:
            i += 1
    return lo + onset_offset


def summarize_releases(analysis: ThrowAnalysis):
    """Per-release-event dicts: time, throwing-hand speed, peak separation,
    elbow angle (at release and at max cocking), trunk lean, stride length
    (peak foot separation) around release, and time-to-release.

    "Max cocking" is approximated as the local minimum in wrist speed just
    before release - the brief pause at the top of the throwing motion
    before the arm accelerates forward - since a true external-rotation
    angle needs a forearm axis MediaPipe's body landmarks don't provide.

    "Time to release" is the duration from when the throwing motion first
    starts (the end of the last sustained stretch of near-idle wrist speed
    before release - see _find_movement_onset) to release itself, similar
    in spirit to "time to throw" in broadcast/analytics football stats,
    except measured from the body starting to move rather than from a snap.
    """
    events = []
    lookback_frames = max(1, int(round(1.0 * analysis.fps)))

    for idx in analysis.release_frames:
        t = analysis.speed_timestamps[idx]
        speed_mps = analysis.wrist_speed[idx]
        release_ts_i = int(np.argmin(np.abs(analysis.timestamps_s - t)))

        window_mask = (analysis.timestamps_s >= t - 1.0) & (analysis.timestamps_s <= t)
        sep_window = analysis.separation_deg[window_mask]
        sep_ts_window = analysis.timestamps_s[window_mask]
        peak_sep_i = int(np.argmax(sep_window))

        stride_window = analysis.foot_separation_m[window_mask]

        lo = max(0, idx - lookback_frames)
        cocking_window = analysis.wrist_speed[lo:idx]
        if len(cocking_window) > 0:
            cocking_t = analysis.speed_timestamps[lo + int(np.argmin(cocking_window))]
            cocking_ts_i = int(np.argmin(np.abs(analysis.timestamps_s - cocking_t)))
            elbow_angle_max_cocking_deg = float(analysis.elbow_angle_deg[cocking_ts_i])
        else:
            elbow_angle_max_cocking_deg = None

        onset_idx = _find_movement_onset(analysis.wrist_speed, idx, analysis.fps)
        onset_t = float(analysis.speed_timestamps[onset_idx])

        events.append(
            {
                "time_s": t,
                "speed_mps": speed_mps,
                "speed_mph": speed_mps * 2.237,
                "peak_separation_deg": sep_window[peak_sep_i],
                "separation_lead_ms": (t - sep_ts_window[peak_sep_i]) * 1000,
                "elbow_angle_release_deg": float(analysis.elbow_angle_deg[release_ts_i]),
                "elbow_angle_max_cocking_deg": elbow_angle_max_cocking_deg,
                "trunk_lean_release_deg": float(analysis.trunk_lean_deg[release_ts_i]),
                "stride_length_m": float(stride_window.max()) if len(stride_window) else None,
                "movement_onset_time_s": onset_t,
                "time_to_release_ms": (t - onset_t) * 1000,
            }
        )
    return events


def summarize_consistency(events):
    """Mean/stddev/coefficient-of-variation across all release events in a
    clip - useful for seeing how repeatable a QB's mechanics are across reps,
    not just the numbers for any one throw. Returns None with fewer than 2
    events (nothing to compare)."""
    if len(events) < 2:
        return None

    keys = [
        "speed_mps",
        "peak_separation_deg",
        "elbow_angle_release_deg",
        "trunk_lean_release_deg",
        "stride_length_m",
        "time_to_release_ms",
    ]
    summary = {}
    for key in keys:
        values = [e[key] for e in events if e.get(key) is not None]
        if len(values) < 2:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values))
        summary[key] = {
            "mean": mean,
            "std": std,
            "cv_pct": (std / mean * 100) if mean else None,
        }
    return summary


def render_annotated_video(
    video_path,
    output_path,
    analysis: ThrowAnalysis,
    model_path=MODEL_PATH_DEFAULT,
    flash_frames=6,
    web_safe=True,
    on_progress=None,
):
    """Write an annotated video: skeleton + elbow angle + wrist speed +
    hip-shoulder separation + a brief RELEASE flash at each detected event.

    Also opens a live preview window if a display is available. When
    web_safe (the default), the output is transcoded to H.264 afterwards so
    it plays inline in browsers - cv2.VideoWriter's own 'mp4v' output often won't.

    on_progress(frame_count, total_frames) is called after every frame, same
    as analyze_video - total_frames may be None if unreliable, and does not
    include the (fast) transcode step.
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (analysis.frame_width, analysis.frame_height)
        )

        frame_count = 0
        while cap.isOpened():
            ret, frame = _read_resized(cap, analysis.frame_width, analysis.frame_height)
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
            if on_progress:
                on_progress(frame_count, total_frames)

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


# --- Ball tracking (best-effort) --------------------------------------------
#
# Everything above measures the throwing HAND, not the ball - wrist speed is
# a proxy for release speed, not the real thing. This section tries to track
# the actual football for a short window after each detected release, to get
# a real speed and launch angle. It's a lightweight classical-CV detector
# (frame differencing + contour filtering in a small region ahead of the
# hand), not a trained object detector - it works best against a relatively
# uncluttered background and can simply fail to find the ball, especially in
# a busy scene. A release event with no confident detection streak is left
# out of the result rather than reported with made-up numbers.


@dataclass
class BallRelease:
    frame_idx: int
    speed_mps: float
    speed_mph: float
    angle_deg: float  # from horizontal, image plane; positive = upward
    positions: list  # [(frame_idx, x_px, y_px), ...] detections used for the fit


def _segment_scale_m_per_px(image_lm, world_lm, frame_width, frame_height, idx_a, idx_b):
    """meters-per-pixel implied by one body segment: its real length (from
    the metric-scale world landmarks) divided by its length in this frame's
    pixel space (from the normalized image landmarks)."""
    img_a = np.array([image_lm[idx_a].x * frame_width, image_lm[idx_a].y * frame_height])
    img_b = np.array([image_lm[idx_b].x * frame_width, image_lm[idx_b].y * frame_height])
    px_dist = np.linalg.norm(img_a - img_b)
    if px_dist < 5:  # too foreshortened by this camera angle to be reliable
        return None
    m_dist = np.linalg.norm(_as_vec(world_lm[idx_a]) - _as_vec(world_lm[idx_b]))
    return float(m_dist / px_dist)


def _frame_scale_m_per_px(image_lm, world_lm, frame_width, frame_height):
    """meters-per-pixel for this frame: median over a few body segments with
    known real length, so any single foreshortened segment doesn't skew it."""
    candidates = []
    for idx_a, idx_b in ((L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP), (R_SHOULDER, R_HIP)):
        scale = _segment_scale_m_per_px(image_lm, world_lm, frame_width, frame_height, idx_a, idx_b)
        if scale is not None:
            candidates.append(scale)
    return float(np.median(candidates)) if candidates else None


def _detect_ball_in_roi(prev_gray, gray, roi):
    """Find the most ball-like moving blob inside roi=(x0,y0,x1,y1) via frame
    differencing. Returns a pixel (x, y) centroid, or None."""
    x0, y0, x1, y1 = roi
    prev_crop = prev_gray[y0:y1, x0:x1]
    crop = gray[y0:y1, x0:x1]
    if prev_crop.size == 0 or crop.size == 0:
        return None

    diff = cv2.absdiff(prev_crop, crop)
    _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bw / bh if bh else 0
        if not (0.3 <= aspect <= 3.0):  # reject long thin motion smears (limbs, edges)
            continue
        if area > best_area:
            best_area = area
            best = (x0 + bx + bw / 2, y0 + by + bh / 2)
    return best


def track_ball_release(video_path, analysis: ThrowAnalysis, model_path=MODEL_PATH_DEFAULT, search_frames=10, roi_radius_px=140):
    """Best-effort tracking of the football for a short window after each
    detected release. Returns a list of BallRelease, one per release event
    where the ball was confidently found (may be shorter than
    analysis.release_frames, or empty).
    """
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    image_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open this file as a video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    w, h = analysis.frame_width, analysis.frame_height

    releases = []
    with PoseLandmarker.create_from_options(image_options) as landmarker:
        for release_frame in sorted(analysis.release_video_frames):
            # One seek per release event, not two: the search loop below diffs
            # forward from this frame instead of re-seeking to release_frame-1
            # (a second seek is needless overhead, and can be genuinely slow on
            # long-GOP video where a seek means decoding from the prior keyframe).
            cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame)
            ret, ref_frame = _read_resized(cap, w, h)
            if not ret:
                continue

            rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            if not result.pose_landmarks or not result.pose_world_landmarks:
                continue
            image_lm = result.pose_landmarks[0]
            world_lm = result.pose_world_landmarks[0]

            scale = _frame_scale_m_per_px(image_lm, world_lm, w, h)
            if scale is None:
                continue

            wrist_px = np.array([image_lm[R_WRIST].x * w, image_lm[R_WRIST].y * h])
            elbow_px = np.array([image_lm[R_ELBOW].x * w, image_lm[R_ELBOW].y * h])
            throw_dir = wrist_px - elbow_px
            norm = np.linalg.norm(throw_dir)
            throw_dir = throw_dir / norm if norm > 1e-6 else np.array([1.0, 0.0])

            prev_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

            detections = []
            search_center = wrist_px.copy()
            for offset in range(search_frames):
                ret, frame = _read_resized(cap, w, h)
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                search_center = search_center + throw_dir * roi_radius_px * 0.5
                x0 = int(max(0, search_center[0] - roi_radius_px))
                y0 = int(max(0, search_center[1] - roi_radius_px))
                x1 = int(min(w, search_center[0] + roi_radius_px))
                y1 = int(min(h, search_center[1] + roi_radius_px))

                hit = _detect_ball_in_roi(prev_gray, gray, (x0, y0, x1, y1))
                if hit is not None:
                    detections.append((release_frame + 1 + offset, hit[0], hit[1]))
                    search_center = np.array(hit)

                prev_gray = gray

            if len(detections) < 3:
                continue

            # A frame-differencing detector in a busy scene (other players,
            # camera shake, grass texture) will happily "find" something in
            # the ROI most frames - it just won't be the ball consistently.
            # Reject tracks that aren't plausibly one real object in real
            # flight: real net motion, roughly toward where the QB threw,
            # and frame-to-frame speed/direction that doesn't zigzag.
            positions_px = np.array([(x, y) for _, x, y in detections], dtype=float)
            net_disp = positions_px[-1] - positions_px[0]
            if np.linalg.norm(net_disp) < roi_radius_px * 0.5 or np.dot(net_disp, throw_dir) <= 0:
                continue

            step_vecs = np.diff(positions_px, axis=0)
            step_dt = np.diff([f for f, _, _ in detections]) / fps
            step_speeds_px = np.linalg.norm(step_vecs, axis=1) / step_dt
            if step_speeds_px.mean() == 0 or step_speeds_px.std() / step_speeds_px.mean() > 0.75:
                continue  # too erratic frame-to-frame to trust as one tracked object

            step_dirs = step_vecs / np.linalg.norm(step_vecs, axis=1, keepdims=True)
            cos_sims = np.sum(step_dirs[:-1] * step_dirs[1:], axis=1)
            if len(cos_sims) > 0 and cos_sims.mean() < 0.2:
                continue  # consecutive steps reverse direction too much to be real flight

            speeds, angles = [], []
            for (f0, x0, y0), (f1, x1, y1) in zip(detections, detections[1:]):
                dt = (f1 - f0) / fps
                if dt <= 0:
                    continue
                dx_m, dy_m = (x1 - x0) * scale, -(y1 - y0) * scale  # image y grows downward
                speeds.append(np.hypot(dx_m, dy_m) / dt)
                angles.append(np.degrees(np.arctan2(dy_m, dx_m)))

            if not speeds:
                continue

            releases.append(
                BallRelease(
                    frame_idx=release_frame,
                    speed_mps=float(np.median(speeds)),
                    speed_mph=float(np.median(speeds)) * 2.237,
                    angle_deg=float(np.median(angles)),
                    positions=detections,
                )
            )

    cap.release()
    return releases

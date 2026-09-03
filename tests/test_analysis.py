import cv2
import numpy as np
import pytest

from analysis import (
    L_EAR,
    L_EYE,
    NOSE,
    R_EAR,
    R_EYE,
    _detect_ball_in_roi,
    _find_movement_onset,
    _target_size,
    angle_between,
    blur_faces,
    find_peaks,
    smooth,
    transverse_angle,
)


def test_smooth_preserves_length():
    values = [1, 2, 3, 4, 5, 4, 3, 2, 1]
    result = smooth(values, window=3)
    assert len(result) == len(values)


def test_smooth_reduces_noise():
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 4 * np.pi, 200))
    noisy = clean + rng.normal(scale=0.5, size=clean.shape)
    smoothed = smooth(noisy, window=7)
    assert np.std(smoothed - clean) < np.std(noisy - clean)


def test_smooth_short_input_returned_unchanged():
    values = [1.0, 2.0]
    result = smooth(values, window=5)
    np.testing.assert_array_equal(result, np.array(values))


class _Point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


@pytest.mark.parametrize(
    "p_from, p_to, expected_deg",
    [
        (_Point(0, 0, 0), _Point(1, 0, 0), 0.0),
        (_Point(0, 0, 0), _Point(0, 0, 1), 90.0),
        (_Point(0, 0, 0), _Point(-1, 0, 0), 180.0),
        (_Point(0, 0, 0), _Point(0, 0, -1), -90.0),
    ],
)
def test_transverse_angle(p_from, p_to, expected_deg):
    assert transverse_angle(p_from, p_to) == pytest.approx(expected_deg)


def test_find_peaks_finds_distinct_local_maxima():
    values = np.array([0, 1, 5, 1, 0, 0, 1, 6, 1, 0], dtype=float)
    peaks = find_peaks(values, min_distance=2, min_height=3)
    assert peaks == [2, 7]


def test_find_peaks_respects_min_distance():
    # Two close peaks - only the taller one should survive within min_distance.
    values = np.array([0, 4, 3, 5, 0], dtype=float)
    peaks = find_peaks(values, min_distance=3, min_height=1)
    assert peaks == [3]


def test_find_peaks_respects_min_height():
    values = np.array([0, 1, 0, 1, 0], dtype=float)
    peaks = find_peaks(values, min_distance=1, min_height=5)
    assert peaks == []


@pytest.mark.parametrize(
    "v1, v2, expected_deg",
    [
        ((1, 0, 0), (1, 0, 0), 0.0),
        ((1, 0, 0), (0, 1, 0), 90.0),
        ((1, 0, 0), (-1, 0, 0), 180.0),
    ],
)
def test_angle_between(v1, v2, expected_deg):
    assert angle_between(np.array(v1), np.array(v2)) == pytest.approx(expected_deg)


def test_detect_ball_in_roi_tracks_a_moving_blob():
    frame1 = np.zeros((200, 200), dtype=np.uint8)
    frame2 = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(frame1, (50, 50), 8, 255, -1)
    cv2.circle(frame2, (70, 60), 8, 255, -1)

    hit = _detect_ball_in_roi(frame1, frame2, (0, 0, 200, 200))
    assert hit is not None
    assert hit == pytest.approx((70.5, 60.5), abs=2)


def test_detect_ball_in_roi_no_motion_returns_none():
    frame = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(frame, (50, 50), 8, 255, -1)
    assert _detect_ball_in_roi(frame, frame, (0, 0, 200, 200)) is None


@pytest.mark.parametrize(
    "width, height, max_width, expected",
    [
        (640, 360, 960, (640, 360)),  # already under the cap - unchanged
        (960, 540, 960, (960, 540)),  # exactly at the cap - unchanged
        (1920, 1080, 960, (960, 540)),  # downscaled, aspect ratio preserved
    ],
)
def test_target_size(width, height, max_width, expected):
    assert _target_size(width, height, max_width) == expected


def test_find_movement_onset_finds_end_of_idle_stretch():
    fps = 30
    idle = np.full(30, 0.2)  # 1s at rest
    ramp = np.linspace(0.2, 5.0, 20)  # ramps up into the release
    speed = np.concatenate([idle, ramp])
    release_idx = len(speed) - 1

    onset = _find_movement_onset(speed, release_idx, fps)
    assert 28 <= onset <= 32  # right around the idle->moving transition


class _Landmark:
    def __init__(self, x, y, visibility=1.0):
        self.x, self.y, self.visibility = x, y, visibility


def _face_landmarks(cx, cy, w, h, spread=0.03):
    """Fake pose landmarks with a face cluster at normalized (cx, cy)."""
    lm = [_Landmark(0, 0)] * 9
    lm[NOSE] = _Landmark(cx, cy)
    lm[L_EYE] = _Landmark(cx + spread, cy - spread)
    lm[R_EYE] = _Landmark(cx - spread, cy - spread)
    lm[L_EAR] = _Landmark(cx + spread * 2, cy)
    lm[R_EAR] = _Landmark(cx - spread * 2, cy)
    return lm


def test_blur_faces_blurs_the_head_region_via_landmarks():
    w, h = 300, 300
    # A noisy/high-frequency image so blur is easy to detect via variance.
    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)

    landmarks = _face_landmarks(cx=0.5, cy=0.3, w=w, h=h)
    before_var = canvas[60:120, 120:180].var()

    blurred = blur_faces(canvas.copy(), landmarks)
    after_var = blurred[60:120, 120:180].var()

    assert after_var < before_var * 0.5  # substantially smoothed where the face was


def test_blur_faces_no_landmarks_and_no_face_leaves_image_unchanged():
    w, h = 200, 200
    canvas = np.full((h, w, 3), 127, dtype=np.uint8)  # flat color, no face for the cascade to find
    result = blur_faces(canvas.copy(), None)
    np.testing.assert_array_equal(result, canvas)


def test_find_movement_onset_ignores_a_brief_pause_mid_motion():
    fps = 30
    idle = np.full(30, 0.2)  # true rest
    windup = np.linspace(0.2, 3.0, 15)  # winding up
    pause = np.full(5, 0.5)  # a brief pause mid-motion - still well above true rest
    release = np.linspace(0.5, 5.0, 10)
    speed = np.concatenate([idle, windup, pause, release])
    release_idx = len(speed) - 1

    onset = _find_movement_onset(speed, release_idx, fps)
    # Onset should land at the end of the *true* idle stretch (~30), not the
    # brief mid-motion pause (~50) - that's the whole point of using a
    # clip-wide baseline instead of a threshold relative to this throw's peak.
    assert onset < 40

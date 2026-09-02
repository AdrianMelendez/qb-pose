import cv2
import numpy as np
import pytest

from analysis import _detect_ball_in_roi, angle_between, find_peaks, smooth, transverse_angle


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

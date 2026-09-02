from pathlib import Path

import pytest

from analysis import (
    MODEL_PATH_DEFAULT,
    analyze_video,
    summarize_consistency,
    summarize_releases,
    track_ball_release,
)

VIDEO_PATH = "qb-throw.mp4"

pytestmark = pytest.mark.skipif(
    not Path(MODEL_PATH_DEFAULT).exists() or not Path(VIDEO_PATH).exists(),
    reason="requires the bundled model + sample video",
)


def test_analyze_video_detects_releases():
    analysis = analyze_video(VIDEO_PATH, model_path=MODEL_PATH_DEFAULT, view="back")
    assert len(analysis.timestamps_s) > 100
    assert len(analysis.release_frames) >= 1

    events = summarize_releases(analysis)
    assert len(events) == len(analysis.release_frames)
    for event in events:
        assert event["speed_mps"] > 0
        assert 0 <= event["peak_separation_deg"] <= 180
        assert 0 <= event["elbow_angle_release_deg"] <= 180
        assert 0 <= event["trunk_lean_release_deg"] <= 180
        assert event["stride_length_m"] >= 0


def test_analyze_video_rejects_unreadable_file(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"not actually a video")
    with pytest.raises(ValueError):
        analyze_video(str(bogus), model_path=MODEL_PATH_DEFAULT, view="back")


def test_summarize_consistency_across_reps():
    analysis = analyze_video(VIDEO_PATH, model_path=MODEL_PATH_DEFAULT, view="back")
    events = summarize_releases(analysis)
    consistency = summarize_consistency(events)

    assert len(events) >= 2  # this clip has several reps
    assert consistency is not None
    for stat in consistency.values():
        assert stat["std"] >= 0
        assert stat["mean"] > 0


def test_summarize_consistency_needs_at_least_two_events():
    assert summarize_consistency([{"speed_mps": 5.0}]) is None
    assert summarize_consistency([]) is None


def test_track_ball_release_only_reports_confident_tracks():
    # Ball tracking is best-effort classical CV; on this back-view clip (busy
    # background, ball moving mostly in depth rather than laterally) it's
    # expected to find zero confident tracks rather than report noise. This
    # asserts it doesn't crash and, whatever it returns, every entry is
    # physically plausible (not a huge speed spike from mismatched blobs).
    analysis = analyze_video(VIDEO_PATH, model_path=MODEL_PATH_DEFAULT, view="back")
    releases = track_ball_release(VIDEO_PATH, analysis, model_path=MODEL_PATH_DEFAULT)
    for ball in releases:
        assert 0 < ball.speed_mps < 45  # ~100mph ceiling, well above any human throw
        assert -180 <= ball.angle_deg <= 180

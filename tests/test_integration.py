from pathlib import Path

import pytest

from analysis import MODEL_PATH_DEFAULT, analyze_video, summarize_releases

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


def test_analyze_video_rejects_unreadable_file(tmp_path):
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"not actually a video")
    with pytest.raises(ValueError):
        analyze_video(str(bogus), model_path=MODEL_PATH_DEFAULT, view="back")

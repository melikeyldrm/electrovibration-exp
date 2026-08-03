"""Tests for the physical calibration of the participant display.

The point of the calibration is that the participant's finger travels the
same physical distance, and the pacing cue asks for the same physical speed,
no matter how large the window is. That property is what these tests check -
they do not check that the calibration numbers themselves are correct, which
only a ruler can establish.
"""

import os

import pytest

from evexp.hardware.screen import ScreenCalibration

# A widget cannot be constructed without a QApplication, and CI has no
# display, so Qt is asked for its headless backend before it is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def cal():
    """A 1920x1080 panel measuring 476x268 mm: about 4.03 px per mm."""
    return ScreenCalibration(
        width_px=1920, height_px=1080,
        width_mm=476.0, height_mm=268.0,
    )


# --- ScreenCalibration -----------------------------------------------------

def test_conversion_factors(cal):
    assert cal.mm_per_px_x == pytest.approx(476.0 / 1920)
    assert cal.px_per_mm_x == pytest.approx(1920 / 476.0)


def test_conversions_are_inverse(cal):
    for mm in (0.0, 1.0, 37.5, 100.0):
        assert cal.px_to_mm_x(cal.mm_to_px_x(mm)) == pytest.approx(mm)


def test_rejects_nonsense_geometry():
    with pytest.raises(ValueError):
        ScreenCalibration(width_px=0, height_px=1080,
                          width_mm=476.0, height_mm=268.0)
    with pytest.raises(ValueError):
        ScreenCalibration(width_px=1920, height_px=1080,
                          width_mm=-476.0, height_mm=268.0)


def test_square_pixels_produce_no_warning(cal):
    assert cal.warnings() == []


def test_non_square_pixels_are_flagged():
    # Height mistyped: 26.8 mm instead of 268 mm.
    bad = ScreenCalibration(width_px=1920, height_px=1080,
                            width_mm=476.0, height_mm=26.8)
    assert any("non-square" in w for w in bad.warnings())


def test_from_config_reports_missing_keys():
    with pytest.raises(KeyError):
        ScreenCalibration.from_config({"screen_width_px": 1920})


# --- CueTrack geometry -----------------------------------------------------
# This is the regression the calibration exists for: the track used to be
# drawn as "window width minus a fixed margin", so the same finger movement
# mapped to a different distance in millimetres depending on the window size.

@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def make_track(qapp, cal, width_px, travel_mm=100.0, cue_speed_mm_s=50.0):
    from evexp.ui.participant_window import CueTrack

    track = CueTrack(cal, position_source=None, travel_mm=travel_mm,
                     cue_speed_mm_s=cue_speed_mm_s)
    track.resize(width_px, 400)
    return track


def test_track_length_is_physical_not_proportional(qapp, cal):
    """100 mm of travel occupies the same pixel length in any window."""
    expected_px = cal.mm_to_px_x(100.0)
    for width in (1100, 1600, 1920):
        track = make_track(qapp, cal, width)
        assert track._track_length_px() == pytest.approx(expected_px)


def test_track_is_centred(qapp, cal):
    track = make_track(qapp, cal, 1920)
    x0, x1, _ = track._track_bounds()
    assert x0 == pytest.approx(1920 - x1)


def test_same_pixel_sweep_gives_same_millimetres(qapp, cal):
    """The regression test proper.

    A finger moving a fixed number of pixels must report the same distance in
    millimetres regardless of window size. Measured as a delta rather than an
    absolute position, because the track is centred and so starts at a
    different x in each window.
    """
    readings = []
    for width in (1100, 1600, 1920):
        track = make_track(qapp, cal, width)
        x0, _, _ = track._track_bounds()
        readings.append(track._px_to_mm(x0 + 200) - track._px_to_mm(x0))

    assert readings[0] == pytest.approx(cal.px_to_mm_x(200))
    for value in readings[1:]:
        assert value == pytest.approx(readings[0])


def test_position_round_trips_through_pixels(qapp, cal):
    track = make_track(qapp, cal, 1600)
    for mm in (0.0, 25.0, 60.0, 100.0):
        assert track._px_to_mm(track._mm_to_px(mm)) == pytest.approx(mm)


def test_position_is_clamped_to_the_track(qapp, cal):
    track = make_track(qapp, cal, 1600)
    x0, x1, _ = track._track_bounds()
    assert track._px_to_mm(x0 - 500) == 0.0
    assert track._px_to_mm(x1 + 500) == pytest.approx(100.0)


def test_narrow_window_falls_back_and_warns(qapp, cal, capsys):
    """A window too narrow for the full travel shrinks the track, loudly."""
    track = make_track(qapp, cal, 200)   # ~403 px needed, 120 px available
    length = track._track_length_px()

    assert length < cal.mm_to_px_x(100.0)
    assert "NOT physically calibrated" in capsys.readouterr().out

    # The display stays self-consistent even in the degraded case.
    assert track._px_to_mm(track._mm_to_px(50.0)) == pytest.approx(50.0)


# --- Pacing cue ------------------------------------------------------------
# The cue is driven by speed rather than by the interval duration, so that
# interval length and sliding speed can be set independently.

def test_cue_travels_at_the_requested_speed(qapp, cal):
    track = make_track(qapp, cal, 1920, cue_speed_mm_s=50.0)
    assert track._cue_position_mm(0.0) == pytest.approx(0.0)
    assert track._cue_position_mm(1.0) == pytest.approx(50.0)
    assert track._cue_position_mm(2.0) == pytest.approx(100.0)


def test_cue_reverses_at_the_end_of_the_track(qapp, cal):
    track = make_track(qapp, cal, 1920, cue_speed_mm_s=50.0)
    # 2 s out, then back: at 3 s it has retraced 50 mm.
    assert track._cue_position_mm(3.0) == pytest.approx(50.0)
    assert track._cue_position_mm(4.0) == pytest.approx(0.0)


def test_cue_stays_on_the_track_at_every_speed(qapp, cal):
    for speed in (20.0, 50.0, 80.0):
        track = make_track(qapp, cal, 1920, cue_speed_mm_s=speed)
        for step in range(400):
            position = track._cue_position_mm(step * 0.025)
            assert 0.0 <= position <= 100.0


def test_cue_speed_is_independent_of_window_size(qapp, cal):
    """Same elapsed time, same physical position, any window."""
    positions = []
    for width in (1100, 1920):
        track = make_track(qapp, cal, width, cue_speed_mm_s=80.0)
        positions.append(track._cue_position_mm(0.7))
    assert positions[0] == pytest.approx(positions[1])
    assert positions[0] == pytest.approx(56.0)   # 80 mm/s x 0.7 s
"""Tests for the force feedback mapping.

This is pure computation kept separate from Qt on purpose, so the tuning
that will happen after watching a real participant - band width, smoothing
window - can be checked without a display.
"""

import pytest

from evexp.processing.force_feedback import (ForceBands, ForceSmoother,
                                              force_border_px, force_color)
from evexp.ui import theme


@pytest.fixture
def bands():
    return ForceBands(target_n=1.0, full_scale_n=0.5)


# --- ForceBands --------------------------------------------------------

def test_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        ForceBands(target_n=1.0, full_scale_n=0.0)


def test_signed_error_direction(bands):
    assert bands.signed_error(1.3) == pytest.approx(0.3)
    assert bands.signed_error(0.7) == pytest.approx(-0.3)


def test_normalized_error_is_clamped(bands):
    assert bands.normalized_error(1.0 + 10.0) == pytest.approx(1.0)
    assert bands.normalized_error(1.0 - 10.0) == pytest.approx(-1.0)


def test_normalized_error_at_full_scale(bands):
    assert bands.normalized_error(1.5) == pytest.approx(1.0)
    assert bands.normalized_error(0.5) == pytest.approx(-1.0)


# --- force_color: monotonic, single ordered axis ---------------------------

def test_on_target_is_the_target_color(bands):
    assert force_color(1.0, bands) == theme.FORCE_TARGET


def test_no_contact_is_the_neutral_color(bands):
    assert force_color(None, bands) == theme.FORCE_UNKNOWN


def test_too_light_moves_toward_low_color(bands):
    at_edge = force_color(0.5, bands)     # full scale under target
    assert at_edge.lower() == theme.FORCE_LOW.lower()


def test_too_hard_moves_toward_high_color(bands):
    at_edge = force_color(1.5, bands)     # full scale over target
    assert at_edge.lower() == theme.FORCE_HIGH.lower()


def test_low_and_high_never_share_a_color(bands):
    """The failure mode this exists to avoid: two wrong directions, one hue."""
    too_light = force_color(0.6, bands)
    too_hard = force_color(1.4, bands)
    assert too_light != too_hard


def test_color_is_monotonic_with_distance_from_target(bands):
    """Getting further from target should never move the color back toward it."""
    def channel_distance(hex_a, hex_b):
        a = tuple(int(hex_a[i:i+2], 16) for i in (1, 3, 5))
        b = tuple(int(hex_b[i:i+2], 16) for i in (1, 3, 5))
        return sum((x - y) ** 2 for x, y in zip(a, b))

    target = force_color(1.0, bands)
    near = force_color(1.1, bands)
    far = force_color(1.3, bands)
    assert channel_distance(target, near) < channel_distance(target, far)


def test_beyond_full_scale_clamps_rather_than_extrapolating(bands):
    assert force_color(1.5, bands) == force_color(3.0, bands)
    assert force_color(0.5, bands) == force_color(-3.0, bands)


# --- force_border_px: second, colour-independent channel -------------------

def test_border_is_thin_on_target(bands):
    assert force_border_px(1.0, bands) == pytest.approx(theme.FORCE_BORDER_MIN_PX)


def test_border_thickens_with_distance_from_target(bands):
    near = force_border_px(1.1, bands)
    far = force_border_px(1.3, bands)
    assert near < far


def test_border_is_symmetric_for_equal_and_opposite_error(bands):
    under = force_border_px(0.8, bands)
    over = force_border_px(1.2, bands)
    assert under == pytest.approx(over)


def test_border_is_thin_when_no_contact(bands):
    assert force_border_px(None, bands) == pytest.approx(theme.FORCE_BORDER_MIN_PX)


def test_border_reaches_max_at_full_scale(bands):
    assert force_border_px(1.5, bands) == pytest.approx(theme.FORCE_BORDER_MAX_PX)
    assert force_border_px(2.5, bands) == pytest.approx(theme.FORCE_BORDER_MAX_PX)


# --- ForceSmoother -----------------------------------------------------

def test_smoother_rejects_a_nonpositive_window():
    with pytest.raises(ValueError):
        ForceSmoother(window_samples=0)


def test_smoother_averages_recent_values():
    smoother = ForceSmoother(window_samples=3)
    smoother.add(1.0)
    smoother.add(2.0)
    assert smoother.add(3.0) == pytest.approx(2.0)


def test_smoother_window_slides():
    smoother = ForceSmoother(window_samples=2)
    smoother.add(0.0)
    smoother.add(10.0)
    assert smoother.add(20.0) == pytest.approx(15.0)   # only last two: 10, 20


def test_smoother_damps_a_single_spike():
    smoother = ForceSmoother(window_samples=5)
    for _ in range(4):
        smoother.add(1.0)
    spiked = smoother.add(5.0)
    assert 1.0 < spiked < 5.0


def test_none_resets_immediately_rather_than_fading():
    """A lifted finger must show up at once, not blend in over the window."""
    smoother = ForceSmoother(window_samples=5)
    for _ in range(5):
        smoother.add(2.0)
    assert smoother.add(None) is None
    assert smoother.add(0.1) == pytest.approx(0.1)   # no memory of the old value


def test_reset_clears_history():
    smoother = ForceSmoother(window_samples=3)
    smoother.add(9.0)
    smoother.add(9.0)
    smoother.reset()
    assert smoother.add(0.2) == pytest.approx(0.2)
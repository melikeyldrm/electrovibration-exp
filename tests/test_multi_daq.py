"""Tests for joining several cards into one acquisition stream.

The behaviour that matters is that channel order and block alignment survive
the join: everything above MultiDaqDevice addresses data by channel name and
assumes every row of a chunk covers the same samples.
"""

import numpy as np
import pytest

from evexp.hardware.mock import MockDAQDevice
from evexp.hardware.multi_daq import MultiDaqDevice


def _card(names, **kwargs):
    return MockDAQDevice(channels=names, realtime=False, **kwargs)


def test_channels_are_concatenated_in_card_order():
    multi = MultiDaqDevice([_card(("fs1_gauge0", "fs1_gauge1")),
                            _card(("fs2_gauge0",)),
                            _card(("current",))])
    assert multi.channels == ("fs1_gauge0", "fs1_gauge1", "fs2_gauge0", "current")


def test_chunk_rows_follow_the_channel_order():
    """A row must belong to the channel its name claims, not a neighbour."""
    multi = MultiDaqDevice([_card(("a", "b")), _card(("c",))])
    multi.start(sample_rate_hz=1000)
    chunk = multi.read_chunk(64)
    assert chunk.data.shape == (3, 64)
    assert chunk.channels == ("a", "b", "c")
    multi.stop()


def test_duplicate_channel_names_are_rejected():
    """Two cards claiming one name would silently shadow each other."""
    with pytest.raises(ValueError):
        MultiDaqDevice([_card(("gauge0",)), _card(("gauge0",))])


def test_no_devices_is_rejected():
    with pytest.raises(ValueError):
        MultiDaqDevice([])


def test_start_unwinds_when_a_later_card_fails():
    """A half-started set would hold its cards open with nothing to close them."""
    class FailingCard(MockDAQDevice):
        def start(self, sample_rate_hz):
            raise RuntimeError("card not present")

    good = _card(("a",))
    multi = MultiDaqDevice([good, FailingCard(channels=("b",))])
    with pytest.raises(RuntimeError):
        multi.start(sample_rate_hz=1000)
    assert not good._running


def test_stop_closes_every_card_even_if_one_fails():
    class FailingStop(MockDAQDevice):
        def stop(self):
            raise RuntimeError("close failed")

    last = _card(("c",))
    multi = MultiDaqDevice([_card(("a",)), FailingStop(channels=("b",)), last])
    multi.start(sample_rate_hz=1000)
    with pytest.raises(RuntimeError):
        multi.stop()
    assert not last._running


def test_mismatched_sample_counts_are_refused():
    """A short read on one card would misalign every row after it."""
    class ShortCard(MockDAQDevice):
        def read_chunk(self, n_samples, timeout_s=10.0):
            return super().read_chunk(max(1, n_samples // 2), timeout_s)

    multi = MultiDaqDevice([_card(("a",)), ShortCard(channels=("b",))])
    multi.start(sample_rate_hz=1000)
    with pytest.raises(RuntimeError, match="different sample counts"):
        multi.read_chunk(64)
    multi.stop()


def test_sample_rate_reported_after_start():
    multi = MultiDaqDevice([_card(("a",))])
    multi.start(sample_rate_hz=10000)
    assert multi.sample_rate_hz == 10000.0
    multi.stop()
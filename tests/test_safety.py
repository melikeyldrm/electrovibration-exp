import numpy as np
import pytest

from evexp.hardware.safety import SafetyLimitExceeded, check_voltage_limit


def test_within_limit_passes():
    check_voltage_limit(np.array([0.5, -1.9, 1.99]), limit_v=2.0)


def test_at_exact_limit_passes():
    check_voltage_limit(np.array([2.0, -2.0]), limit_v=2.0)


def test_over_limit_raises():
    with pytest.raises(SafetyLimitExceeded):
        check_voltage_limit(np.array([0.5, 2.01]), limit_v=2.0)


def test_negative_over_limit_raises():
    with pytest.raises(SafetyLimitExceeded):
        check_voltage_limit(np.array([-2.5]), limit_v=2.0)


def test_default_limit_is_two_volts():
    with pytest.raises(SafetyLimitExceeded):
        check_voltage_limit(np.array([2.5]))


def test_error_message_reports_actual_peak():
    with pytest.raises(SafetyLimitExceeded, match="3.5000"):
        check_voltage_limit(np.array([1.0, -3.5, 0.2]), limit_v=2.0)
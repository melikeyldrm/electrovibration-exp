import random
from dataclasses import dataclass, field
from typing import List

from evexp.hardware.mock import MockDAQDevice
from evexp.psychophysics.staircase import StaircaseController


@dataclass
class TrialResult:
    """Record of a single 2AFC trial."""
    trial_index: int
    applied_voltage: float
    stimulus_interval: int  # 1 or 2 — which interval carried the stimulus
    response_interval: int  # participant's (simulated) answer
    correct: bool
    timestamp: float


class TrialController:
    """Runs a 2AFC experiment loop, driving a staircase with a DAQ-like device.

    The device only needs to implement `set_applied_voltage()` and
    `detection_probability()` for this simulated flow to work; a real
    hardware device would instead read an actual participant response.
    """

    def __init__(self, device: MockDAQDevice, staircase: StaircaseController):
        self.device = device
        self.staircase = staircase
        self.results: List[TrialResult] = []

    def run_trial(self, trial_index: int) -> TrialResult:
        # Randomize which interval carries the stimulus (2AFC design)
        stimulus_interval = random.choice([1, 2])

        voltage = self.staircase.value
        self.device.set_applied_voltage(voltage)

        # Simulate the participant's response using the device's
        # psychometric function; a real system would collect this
        # from a keypress / touchscreen input instead.
        p_correct = self.device.detection_probability()
        correct = random.random() < p_correct
        response_interval = stimulus_interval if correct else (3 - stimulus_interval)

        sample = self.device.read()

        result = TrialResult(
            trial_index=trial_index,
            applied_voltage=voltage,
            stimulus_interval=stimulus_interval,
            response_interval=response_interval,
            correct=correct,
            timestamp=sample.timestamp,
        )
        self.results.append(result)

        # Feed the outcome back into the staircase to get the next voltage
        self.staircase.update(correct)
        return result

    def run_until_done(self, max_trials: int = 300) -> List[TrialResult]:
        with self.device:
            trial_index = 0
            while not self.staircase.finished and trial_index < max_trials:
                self.run_trial(trial_index)
                trial_index += 1
        return self.results
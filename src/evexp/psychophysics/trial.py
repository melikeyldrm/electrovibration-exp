"""Two-interval forced-choice (2IFC - the participant feels two intervals
and says which one had the stimulus, rather than just yes/no) trial
sequencing.


Each interval is preceded by a "place your finger" wait phase (PRE_INTERVAL_WAIT).
"""

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from evexp.hardware.base import StimulusOutput
from evexp.psychophysics.staircase import StaircaseController


class TrialState(Enum):
    """Phases a single trial passes through, in order."""
    READY = auto()
    PRE_INTERVAL_WAIT = auto()  # "place your finger" — shown before each interval
    INTERVAL_1 = auto()
    GAP = auto()
    INTERVAL_2 = auto()
    AWAITING_RESPONSE = auto()
    FINISHED = auto()


@dataclass
class TrialTiming:
    """Timing and pacing parameters shared by every trial in a session.

    interval_s is deliberately not a settable field: the track is a single
    one-way sweep, so interval duration and sliding speed are not
    independent - it's derived from travel/speed instead, so stimulus
    duration varies with speed condition rather than the two ever
    disagreeing.
    """
    pre_interval_wait_s: float = 3.0  
    gap_s: float = 2.0
    cursor_speed_mm_s: float = 50.0
    cursor_travel_mm: float = 100.0

    def __post_init__(self) -> None:
        if self.cursor_speed_mm_s <= 0:
            raise ValueError(
                f"cursor_speed_mm_s must be > 0, got {self.cursor_speed_mm_s!r}"
            )
        if self.cursor_travel_mm <= 0:
            raise ValueError(
                f"cursor_travel_mm must be > 0, got {self.cursor_travel_mm!r}"
            )

    @property
    def interval_s(self) -> float:
        """Interval duration: exactly one one-way sweep across the track."""
        return self.cursor_travel_mm / self.cursor_speed_mm_s


@dataclass
class TrialResult:
    """Record of a single completed 2IFC trial."""
    trial_index: int
    applied_voltage: float
    stimulus_interval: int    # 1 or 2 - which interval carried the stimulus
    response_interval: int    # 1 or 2 - what the participant reported
    correct: bool
    reversal: bool            # did this response produce a staircase reversal
    timestamp: float
    response_time_s: float
    training: bool = False    # True for training trials; staircase not updated

    # Filled in by SessionController after submit_response(), not by
    # Trial2IFC (hardware-free). None means "no force source configured",
    # distinct from a force reading of zero.
    mean_normal_force_n: Optional[float] = None
    std_normal_force_n: Optional[float] = None
    force_in_band_fraction: Optional[float] = None
    cursor_speed_mm_s: Optional[float] = None


class Trial2IFC:
    """Sequences 2IFC trials as a non-blocking state machine.

    READY -> PRE_INTERVAL_WAIT -> INTERVAL_1 -> GAP -> PRE_INTERVAL_WAIT
          -> INTERVAL_2 -> AWAITING_RESPONSE -> READY

    This class contains no GUI code and never waits. The caller drives it:
    start_trial() begins a trial, advance() is called when the current
    phase's timer expires, and submit_response() is called when the
    participant presses 1 or 2. A PyQt front end wires these to QTimer and
    keyPressEvent; the headless simulation below calls them directly.
    """

    def __init__(
        self,
        stimulus: StimulusOutput,
        staircase: StaircaseController,
        timing: Optional[TrialTiming] = None,
        rng: Optional[random.Random] = None,
        training_voltage: float = 2.0,
    ):
        self.stimulus = stimulus
        self.staircase = staircase
        self.timing = timing or TrialTiming()
        self._rng = rng or random.Random()
        self.training_voltage = training_voltage

        self.state = TrialState.READY
        self.trial_index = 0
        self.results: List[TrialResult] = []

        self._stimulus_interval = 1
        self._applied_voltage = 0.0
        self._response_open_t = 0.0
        self._is_training = False
        # Which interval (1 or 2) the current PRE_INTERVAL_WAIT phase is for.
        self._pending_interval = 1

    @property
    def finished(self) -> bool:
        """True once the staircase has reached its stopping criterion."""
        return self.staircase.finished

    @property
    def stimulus_interval(self) -> int:
        """Which interval carries the stimulus in the current trial."""
        return self._stimulus_interval

    @property
    def applied_voltage(self) -> float:
        """Stimulus amplitude used in the current trial, in volts."""
        return self._applied_voltage

    def start_trial(self, training: bool = False) -> float:
        """Begin a trial and enter the pre-interval-1 wait phase.

        Returns the duration of the wait phase in seconds so the caller can
        arm a timer for it. If training=True the staircase value is not used
        and the result will not update the staircase.
        """
        if self.state is not TrialState.READY:
            raise RuntimeError(f"start_trial() called in state {self.state.name}")

        self._is_training = training
        self._applied_voltage = (
            self.training_voltage if training else self.staircase.value
        )
        self._stimulus_interval = self._rng.choice([1, 2])  # choose stimulus ON interval randomly
        self.stimulus.set_amplitude(self._applied_voltage)
        self.stimulus.stimulus_off()  # ensure no stimulus during the wait phase

        self.state = TrialState.PRE_INTERVAL_WAIT
        self._pending_interval = 1
        return self.timing.pre_interval_wait_s # caller waits this long, then calls advance()

    def advance(self) -> Optional[float]:
        """Move to the next phase when the current phase's timer expires.

        Returns the duration of the new phase in seconds, or None once the
        trial is waiting for the participant's response (which is untimed).
        """
        if self.state is TrialState.PRE_INTERVAL_WAIT:
            if self._pending_interval == 1:
                self.state = TrialState.INTERVAL_1
                self._set_stimulus_for_interval(1)
            else:
                self.state = TrialState.INTERVAL_2
                self._set_stimulus_for_interval(2)
            return self.timing.interval_s

        if self.state is TrialState.INTERVAL_1:
            self.stimulus.stimulus_off()
            self.state = TrialState.GAP
            return self.timing.gap_s

        if self.state is TrialState.GAP:
            self.state = TrialState.PRE_INTERVAL_WAIT
            self._pending_interval = 2
            return self.timing.pre_interval_wait_s

        if self.state is TrialState.INTERVAL_2:
            self.stimulus.stimulus_off()
            self.state = TrialState.AWAITING_RESPONSE
            self._response_open_t = time.time()
            return None

        raise RuntimeError(f"advance() called in state {self.state.name}")

    def submit_response(self, response_interval: int) -> TrialResult:
        """Score the participant's answer, update the staircase, end the trial.

        For training trials the staircase is not updated and reversal is always
        False; the state machine returns to READY so the next trial can start.
        """
        if self.state is not TrialState.AWAITING_RESPONSE:
            raise RuntimeError(
                f"submit_response() called in state {self.state.name}"
            )
        if response_interval not in (1, 2):
            raise ValueError("response_interval must be 1 or 2")

        correct = response_interval == self._stimulus_interval

        if self._is_training:
            reversal = False
        else:
            reversals_before = len(self.staircase.reversals)
            self.staircase.update(correct)
            reversal = len(self.staircase.reversals) > reversals_before

        result = TrialResult(
            trial_index=self.trial_index,
            applied_voltage=self._applied_voltage,
            stimulus_interval=self._stimulus_interval,
            response_interval=response_interval,
            correct=correct,
            reversal=reversal,
            timestamp=time.time(),
            response_time_s=time.time() - self._response_open_t,
            training=self._is_training,
        )

        self.results.append(result)
        if not self._is_training:
            self.trial_index += 1

        if self._is_training:
            # Training trials never advance the staircase; always return to READY.
            self.state = TrialState.READY
        else:
            self.state = (
                TrialState.FINISHED if self.staircase.finished else TrialState.READY
            )

        return result

    def _set_stimulus_for_interval(self, interval: int) -> None:
        if interval == self._stimulus_interval:
            self.stimulus.stimulus_on()
        else:
            self.stimulus.stimulus_off()

    def voltages_over_trials(self) -> List[float]:
        """Voltages for real trials only, in order, for convergence plotting."""
        return [r.applied_voltage for r in self.results if not r.training]


class SimulatedRunner:
    """Runs a full session headlessly against a simulated observer.

    Used for tests and for producing convergence plots without a participant.
    Phase durations are not waited out; transitions fire immediately so a
    session that would take 20 minutes with a person runs in milliseconds.
    Training trials are skipped: the simulated observer needs no warm-up.
    """

    def __init__(
        self,
        trial: Trial2IFC,
        true_threshold: float,
        slope: float = 0.15,
        rng: Optional[random.Random] = None,
    ):
        self.trial = trial
        self.true_threshold = true_threshold
        self.slope = slope
        self._rng = rng or random.Random()

    def _observer_response(self, voltage: float, stimulus_interval: int) -> int:
        """Pick an interval using a logistic psychometric function."""
        p_detect = 1.0 / (
            1.0 + math.exp(-self.slope * (voltage - self.true_threshold))
        )
        p_correct = 0.5 + 0.5 * p_detect
        if self._rng.random() < p_correct:
            return stimulus_interval
        return 3 - stimulus_interval

    def run_until_done(self, max_trials: int = 300) -> List[TrialResult]:
        while not self.trial.finished and self.trial.trial_index < max_trials:
            self.trial.start_trial(training=False)
            # Advance through every timed phase (wait/interval/gap/wait/interval)
            # until the trial reaches the untimed AWAITING_RESPONSE phase,
            # signalled by advance() returning None. Robust to the exact
            # number of phases in the state machine.
            while self.trial.advance() is not None:
                pass
            response = self._observer_response(
                self.trial.applied_voltage, self.trial.stimulus_interval
            )
            self.trial.submit_response(response)
        return self.trial.results
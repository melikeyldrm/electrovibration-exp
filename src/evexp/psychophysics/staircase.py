import math
from dataclasses import dataclass
from typing import List, Literal


@dataclass
class StaircaseConfig:
    """Configuration parameters for an adaptive staircase."""

    start_value: float
    step_sizes: List[float]          # Step size after each reversal.
    n_reversals_to_stop: int
    rule: Literal["1up2down", "3down1up"] = "1up2down"
    min_value: float = 0.0
    max_value: float = 1e9
    max_consecutive_wrong_at_ceiling: int = 5  # Session aborts if this many consecutive wrong answers occur at max_value.
                                                # Set to 0 to disable. Literature (Vardar & Kuchenbecker 2021) uses 3;
    domain: Literal["linear", "db"] = "linear"
    # If True (default), the correct-answer streak resets to 0 on any wrong
    # answer, i.e. the required correct answers must be consecutive.
    # If False, a wrong answer still triggers an immediate step (direction="up")
    # but does NOT reset the correct streak — matches "not necessarily
    # consecutive" wording in Vuik/Pool/Kenanoglu/Vardar 2024/2025.
    consecutive_required: bool = True

    # which 5 reversal ?
    require_fine_stage_reversals: bool = False

    def __post_init__(self):
        if self.domain == "db" and self.min_value <= 0:
            raise ValueError(
                "min_value must be > 0 when domain='db' (log(0) is undefined). "
                "Pick a small positive floor, e.g. 0.01."
            )


def to_db(value: float) -> float:
    """Convert a linear peak voltage to dB: 20*log10(Vp)."""
    return 20.0 * math.log10(value)


def from_db(value_db: float) -> float:
    """Convert a dB value back to linear peak voltage."""
    return 10.0 ** (value_db / 20.0)


class StaircaseController:
    """Adaptive staircase controller supporting 1-up/2-down and 3-down/1-up rules,
    in either linear or dB step domain."""

    def __init__(self, config: StaircaseConfig):
        self.cfg = config
        self.value = config.start_value
        self._correct_streak = 0
        self._last_direction: str | None = None
        self._step_idx = 0
        self.reversals: List[float] = []
        self.history: List[dict] = []
        self.finished = False
        self._consecutive_wrong_at_ceiling = 0
        self.aborted = False

    @property
    def step_size(self) -> float:
        """Return the current step size (in the configured domain: linear or dB)."""
        idx = min(self._step_idx, len(self.cfg.step_sizes) - 1)
        return self.cfg.step_sizes[idx]

    def _apply_step(self, direction: str) -> float:
        """Apply the current step to self.value, in linear or dB domain, then clamp."""
        signed_step = self.step_size if direction == "up" else -self.step_size

        if self.cfg.domain == "db":
            new_value = from_db(to_db(self.value) + signed_step)
        else:
            new_value = self.value + signed_step

        return min(self.cfg.max_value, max(self.cfg.min_value, new_value))

    def update(self, correct: bool) -> float:
        """Update the staircase and return the next stimulus value."""

        if self.finished:
            return self.value

        # Number of correct responses required before stepping down.
        need = 2 if self.cfg.rule == "1up2down" else 3

        direction = None

        if correct:
            self._correct_streak += 1
            if self._correct_streak >= need:
                direction = "down"
                self._correct_streak = 0
        else:
            direction = "up"
            if self.cfg.consecutive_required:
                self._correct_streak = 0
            # else: wrong answer triggers an immediate step but does not
            # reset the cumulative correct count (non-consecutive rule).

        if direction is not None:

            # A reversal occurs when the staircase changes direction.
            if self._last_direction and direction != self._last_direction:
                self.reversals.append(self.value)
                self._step_idx += 1

            self._last_direction = direction
            self.value = self._apply_step(direction)

        # Safety abort check: if the participant is repeatedly wrong at the
        # voltage ceiling, continuing serves no purpose and wastes their time.
        if not correct and self.value >= self.cfg.max_value:
            self._consecutive_wrong_at_ceiling += 1
            if (self.cfg.max_consecutive_wrong_at_ceiling > 0
                    and self._consecutive_wrong_at_ceiling
                    >= self.cfg.max_consecutive_wrong_at_ceiling):
                self.aborted = True
                self.finished = True
        else:
            self._consecutive_wrong_at_ceiling = 0

        self.history.append(
            {
                "value": self.value,
                "correct": correct,
                "direction": direction,
            }
        )

        n_coarse_reversals_to_skip = (
            len(self.cfg.step_sizes) - 1
            if self.cfg.require_fine_stage_reversals
            else 0
        )
        if len(self.reversals) >= n_coarse_reversals_to_skip + self.cfg.n_reversals_to_stop:
            self.finished = True
        return self.value

    @property
    def threshold_estimate(self) -> float:
        """Estimate the threshold from the last N reversal points (linear voltage
        mean, where N = n_reversals_to_stop)."""
        n = self.cfg.n_reversals_to_stop
        tail = self.reversals[-n:] or self.reversals
        return sum(tail) / len(tail) if tail else float("nan")

    @property
    def total_reversals_needed(self) -> int:
        """Total reversals before the session stops, including any coarse-stage
        warm-up reversals that don't count toward the threshold average."""
        n_coarse = (len(self.cfg.step_sizes) - 1
                    if self.cfg.require_fine_stage_reversals else 0)
        return n_coarse + self.cfg.n_reversals_to_stop
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
    max_consecutive_wrong_at_ceiling: int = 5 # Session aborts if this many consecutive wrong answers occur at max_value.
                                              # Set to 0 to disable. Literature (Vardar & Kuchenbecker 2021) uses 3;


class StaircaseController:
    """Adaptive staircase controller supporting 1-up/2-down and 3-down/1-up rules."""

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
        """Return the current step size."""
        idx = min(self._step_idx, len(self.cfg.step_sizes) - 1)
        return self.cfg.step_sizes[idx]

    def update(self, correct: bool) -> float:
        """Update the staircase and return the next stimulus value."""

        if self.finished:
            return self.value

        # Number of consecutive correct responses required before stepping down.
        need = 2 if self.cfg.rule == "1up2down" else 3

        direction = None

        if correct:
            self._correct_streak += 1
            if self._correct_streak >= need:
                direction = "down"
                self._correct_streak = 0
        else:
            direction = "up"
            self._correct_streak = 0

        if direction is not None:

            # A reversal occurs when the staircase changes direction.
            if self._last_direction and direction != self._last_direction:
                self.reversals.append(self.value)
                self._step_idx += 1

            self._last_direction = direction

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

            # Update the stimulus value while keeping it within the allowed range.
            self.value += self.step_size if direction == "up" else -self.step_size
            self.value = min(
                self.cfg.max_value,
                max(self.cfg.min_value, self.value),
            )

        self.history.append(
            {
                "value": self.value,
                "correct": correct,
                "direction": direction,
            }
        )

        if len(self.reversals) >= self.cfg.n_reversals_to_stop:
            self.finished = True

        return self.value

    @property
    def threshold_estimate(self) -> float:
        """Estimate the threshold from the last reversal points."""
        tail = self.reversals[-6:] or self.reversals
        return sum(tail) / len(tail) if tail else float("nan")
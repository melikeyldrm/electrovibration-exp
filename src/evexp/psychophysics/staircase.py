from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class StaircaseConfig:
    start_value: float
    step_sizes: List[float]          # örn. [8, 4, 2, 1] — her reversal'da küçülür
    n_reversals_to_stop: int
    rule: Literal["1up2down", "3down1up"] = "1up2down"
    min_value: float = 0.0
    max_value: float = 1e9


class StaircaseController:
    """1-up/2-down (~70.7%) veya 3-down/1-up (~79%) adaptif staircase."""

    def __init__(self, config: StaircaseConfig):
        self.cfg = config
        self.value = config.start_value
        self._correct_streak = 0
        self._last_direction: str | None = None
        self._step_idx = 0
        self.reversals: List[float] = []
        self.history: List[dict] = []
        self.finished = False

    @property
    def step_size(self) -> float:
        idx = min(self._step_idx, len(self.cfg.step_sizes) - 1)
        return self.cfg.step_sizes[idx]

    def update(self, correct: bool) -> float:
        """Bir trial'ın sonucunu ver, bir sonraki stimulus değerini al."""
        if self.finished:
            return self.value

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
            if self._last_direction and direction != self._last_direction:
                self.reversals.append(self.value)
                self._step_idx += 1
            self._last_direction = direction
            self.value += self.step_size if direction == "up" else -self.step_size
            self.value = min(self.cfg.max_value, max(self.cfg.min_value, self.value))

        self.history.append({"value": self.value, "correct": correct, "direction": direction})
        if len(self.reversals) >= self.cfg.n_reversals_to_stop:
            self.finished = True
        return self.value

    @property
    def threshold_estimate(self) -> float:
        tail = self.reversals[-6:] or self.reversals
        return sum(tail) / len(tail) if tail else float("nan")
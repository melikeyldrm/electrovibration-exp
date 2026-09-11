"""Physical calibration of the participant display.

Everything the participant sees is specified in millimetres and converted
to pixels only at draw time, so window resizing can't silently change the
distance/speed values recorded for a trial. Measured once with a ruler,
not from Qt's QScreen.physicalDotsPerInch(), which on Windows usually
reports the logical 96 DPI rather than true panel geometry.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenCalibration:
    """Maps between millimetres on the panel and pixels on the screen.

    width_mm/height_mm refer to the *active display area* of the panel - the
    lit region, not the outer bezel.
    """

    width_px: int
    height_px: int
    width_mm: float
    height_mm: float

    # Pixels are square on essentially every modern panel. If the measured
    # geometry disagrees by more than this, the measurement is more likely to
    # be wrong than the panel is.
    ANISOTROPY_TOLERANCE = 0.02

    def __post_init__(self) -> None:
        for name in ("width_px", "height_px", "width_mm", "height_mm"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(
                    f"ScreenCalibration.{name} must be positive, got {value!r}"
                )

    # --- conversion factors ------------------------------------------------

    @property
    def mm_per_px_x(self) -> float:
        return self.width_mm / self.width_px

    @property
    def mm_per_px_y(self) -> float:
        return self.height_mm / self.height_px

    @property
    def px_per_mm_x(self) -> float:
        return self.width_px / self.width_mm

    @property
    def px_per_mm_y(self) -> float:
        return self.height_px / self.height_mm

    # --- conversions -------------------------------------------------------

    def px_to_mm_x(self, px: float) -> float:
        return px * self.mm_per_px_x

    def mm_to_px_x(self, mm: float) -> float:
        return mm * self.px_per_mm_x

    def px_to_mm_y(self, px: float) -> float:
        return px * self.mm_per_px_y

    def mm_to_px_y(self, mm: float) -> float:
        return mm * self.px_per_mm_y

    # --- diagnostics -------------------------------------------------------

    @property
    def anisotropy(self) -> float:
        """Relative disagreement between the horizontal and vertical scale."""
        return abs(self.mm_per_px_x - self.mm_per_px_y) / self.mm_per_px_x

    def warnings(self) -> list[str]:
        """Non-fatal problems worth printing at session start."""
        issues: list[str] = []
        if self.anisotropy > self.ANISOTROPY_TOLERANCE:
            issues.append(
                f"non-square pixels: {self.mm_per_px_x:.4f} mm/px horizontally "
                f"vs {self.mm_per_px_y:.4f} mm/px vertically "
                f"({self.anisotropy * 100:.1f}% apart). "
                "Check the measured active-area dimensions."
            )
        return issues

    def describe(self) -> str:
        return (
            f"{self.width_px}x{self.height_px} px over "
            f"{self.width_mm:.1f}x{self.height_mm:.1f} mm "
            f"({self.mm_per_px_x:.4f} mm/px, {self.px_per_mm_x:.2f} px/mm)"
        )

    # --- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "ScreenCalibration":
        """Build from the `display:` block of experiment.yaml."""
        try:
            return cls(
                width_px=int(cfg["screen_width_px"]),
                height_px=int(cfg["screen_height_px"]),
                width_mm=float(cfg["screen_width_mm"]),
                height_mm=float(cfg["screen_height_mm"]),
            )
        except KeyError as missing:
            raise KeyError(
                f"display config is missing {missing}; the participant "
                "display cannot be calibrated without it"
            ) from missing
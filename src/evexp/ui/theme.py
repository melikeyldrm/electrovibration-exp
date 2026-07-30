"""Visual style constants for the participant-facing experiment UI.

Kept in one place so the look can be tuned without touching widget logic.

Design rationale: this is a perception experiment, so the priority is
*not* drawing attention. Mid-dark neutral background (not pure black,
which fatigues the eye over long sessions), desaturated accents, no
flashing or sudden luminance changes, and large type so the participant
can read prompts while their hand is on the screen.
"""

# --- Colours ---------------------------------------------------------------

BACKGROUND = "#232326"        # mid-dark neutral; avoids pure-black eye strain
SURFACE = "#2c2c30"           # panels / track background, one step lighter
TEXT_PRIMARY = "#e8e8ea"      # off-white; pure white is harsh on dark bg
TEXT_SECONDARY = "#9a9aa0"    # hints, secondary instructions

# Pacing cursor the participant is asked to follow (circle).
CURSOR = "#7fb8d4"            # desaturated blue; calm, clearly non-signalling

# The participant's own tracked position (square). Deliberately a different
# hue AND a different shape from the cursor, so the two are distinguishable
# without relying on colour discrimination.
PARTICIPANT_MARKER = "#d9a95c"  # muted amber

TRACK = "#3a3a40"             # the 100 mm travel path

# Interval labels. No red/green: the participant answers with the 1 / 2 keys,
# so the intervals are identified by number, which also sidesteps
# red-green colour blindness entirely.
INTERVAL_ACTIVE = "#e8e8ea"
INTERVAL_IDLE = "#5a5a60"

# --- Typography ------------------------------------------------------------

FONT_FAMILY = "Segoe UI"      # present on Windows; falls back gracefully
FONT_SIZE_HUGE = 64           # interval number ("1" / "2")
FONT_SIZE_LARGE = 28          # main prompts ("Place your finger")
FONT_SIZE_BODY = 18           # hints ("Press 1 or 2")

# --- Geometry --------------------------------------------------------------

CURSOR_RADIUS_PX = 22
MARKER_SIZE_PX = 34           # square side; roughly matches cursor diameter
TRACK_HEIGHT_PX = 6
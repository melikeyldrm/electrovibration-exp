"""Visual style constants for the participant-facing experiment UI.

Kept in one place so the look can be tuned without touching widget logic.

Design rationale: this is a perception experiment, so the priority is *not*
drawing attention. Mid-dark neutral background (not pure black, which
fatigues the eye over long sessions), desaturated accents, no flashing or
sudden luminance changes, and large type - the participant reads these
prompts with their hand on the screen, not from a desk.
"""

# --- Colours ---------------------------------------------------------------

BACKGROUND = "#232326"        # mid-dark neutral; avoids pure-black eye strain
TEXT_PRIMARY = "#e8e8ea"      # off-white; pure white is harsh on dark bg
TEXT_SECONDARY = "#9a9aa0"    # hints, secondary instructions
TEXT_EMPHASIS = "#f4f4f6"     # countdown and interval numerals

# Pacing cursor the participant is asked to follow (circle).
CURSOR = "#7fb8d4"            # desaturated blue; calm, clearly non-signalling

# The participant's own tracked position (square). Deliberately a different
# hue AND a different shape from the cursor, so the two stay distinguishable
# without relying on colour discrimination.
PARTICIPANT_MARKER = "#d9a95c"  # muted amber

TRACK = "#3a3a40"             # the 100 mm travel path
START_MARKER = "#5c5c66"      # notch at the start of the track

# --- Typography ------------------------------------------------------------

FONT_FAMILY = "Segoe UI"      # present on Windows; falls back gracefully
FONT_SIZE_NUMERAL = 96        # countdown digits and interval number
FONT_SIZE_LARGE = 34          # main prompts
FONT_SIZE_BODY = 22           # hints and instructions

# --- Geometry --------------------------------------------------------------

# The participant's square sits on top of the pacing circle and must stay
# smaller than it, so that "on pace" reads as the square nested inside the
# circle rather than the two merely overlapping. Both are larger than a
# fingertip is wide, so neither disappears under the hand.
CURSOR_RADIUS_PX = 36         # circle: 72 px across
MARKER_SIZE_PX = 46           # square side; comfortably inside the circle
TRACK_HEIGHT_PX = 8
START_MARKER_HEIGHT_PX = 34   # vertical notch marking the start position

# Fixed heights keep the track from drifting up and down as text changes
# length. The participant physically touches the screen, so the track must
# stay in one place across every phase. Each height is generous relative to
# its font size: Qt clips descenders (g, y, p) if the label is sized tightly.
MESSAGE_AREA_HEIGHT_PX = 100
NUMERAL_AREA_HEIGHT_PX = 190
HINT_AREA_HEIGHT_PX = 80

# --- Experimenter console --------------------------------------------------
# The console usually sits in the same room as the participant, so it is dark
# for the same reason the participant screen is: a bright panel would spill
# light across the setup. Denser and more saturated than the participant UI,
# though - this one is meant to be read closely and scanned quickly.

CONSOLE_BG = "#f0f2f7"
CONSOLE_PANEL = "#ffffff"
CONSOLE_PANEL_ALT = "#f6f8fc"     # alternating table rows
CONSOLE_BORDER = "#d5dae3"
CONSOLE_TEXT = "#1c2330"
CONSOLE_LABEL = "#6b7382"       # small uppercase field labels
CONSOLE_ACCENT = "#0a84d6"       # matches the participant cue colour

CONSOLE_OK = "#1f8a5f"            # on-target speed, reversals
CONSOLE_WARN = "#b0720c"          # drifting
CONSOLE_BAD = "#c0392b"           # wrong answers, far off target

CONSOLE_GRID = "#e2e7ef"
CONSOLE_TRACE = "#0a84d6"
CONSOLE_THRESHOLD = "#1f8a5f"

CONSOLE_FONT = "Segoe UI"
CONSOLE_MONO = "Consolas"
CONSOLE_SIZE_VALUE = 17
CONSOLE_SIZE_LABEL = 9
CONSOLE_SIZE_BODY = 12
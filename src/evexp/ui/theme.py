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

# The participant's own tracked position (square). Its fill communicates
# applied force (see FORCE_* below); TRACK stays fixed.
TRACK = "#3a3a40"             # the 100 mm travel path

# --- Force feedback ----------------------------------------------------
# The square's fill encodes how the participant's applied force compares to
# the target, on a single ordered scale: too little -> on target -> too much.
# Deliberately not "green vs red vs amber for both directions" - two
# out-of-range states sharing the same colour (amber) tell the participant
# something is wrong but not which way to correct, which is worse than no
# colour at all. Blue and red are also the pair least likely to be confused
# under red-green colour blindness, and the built-in border-width cue below
# does not depend on colour perception at all.
FORCE_LOW = "#5b8fc7"          # pressing too lightly: cool blue
FORCE_TARGET = "#5cad6e"       # in the target band: green
FORCE_HIGH = "#c25b52"         # pressing too hard: warm red
FORCE_UNKNOWN = "#6b6b72"      # no contact detected - neutral, not alarming

# Marker border thickens with distance from target, as a second channel that
# does not depend on colour discrimination at all.
FORCE_BORDER_MIN_PX = 0.0
FORCE_BORDER_MAX_PX = 7.0

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
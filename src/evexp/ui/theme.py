"""Visual style constants for the participant-facing experiment UI."""

# --- Colours ---------------------------------------------------------------

BACKGROUND = "#232326"        # mid-dark neutral; avoids pure-black eye strain
TEXT_PRIMARY = "#e8e8ea"      # off-white; pure white is harsh on dark bg
TEXT_SECONDARY = "#9a9aa0"    # hints, secondary instructions
TEXT_EMPHASIS = "#f4f4f6"     # countdown and interval numerals

CURSOR = "#7fb8d4"            # pacing cursor the participant follows (circle)
TRACK = "#3a3a40"             # the 100 mm travel path

# --- Force feedback ----------------------------------------------------
# Square's fill encodes applied force vs. target on one ordered scale.
FORCE_LOW = "#5b8fc7"          # pressing too lightly: cool blue
FORCE_TARGET = "#5cad6e"       # in the target band: green
FORCE_HIGH = "#c25b52"         # pressing too hard: warm red
FORCE_UNKNOWN = "#6b6b72"      # no contact detected

# Marker border thickens with distance from target, as a second channel.
FORCE_BORDER_MIN_PX = 0.0
FORCE_BORDER_MAX_PX = 7.0

# --- Typography ------------------------------------------------------------

FONT_FAMILY = "Segoe UI"      # present on Windows; falls back gracefully
FONT_SIZE_NUMERAL = 96        # countdown digits and interval number
FONT_SIZE_LARGE = 34          # main prompts
FONT_SIZE_BODY = 22           # hints and instructions

# --- Geometry --------------------------------------------------------------

CURSOR_RADIUS_PX = 36         # circle: 72 px across
MARKER_SIZE_PX = 46           # square side; smaller than the circle
TRACK_HEIGHT_PX = 8

# Fixed heights keep the track from drifting as text length changes.
MESSAGE_AREA_HEIGHT_PX = 100
NUMERAL_AREA_HEIGHT_PX = 190
HINT_AREA_HEIGHT_PX = 80

# --- Experimenter console --------------------------------------------------

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

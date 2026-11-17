"""Pinned constants for the submitted rule-based pipeline."""

# Submission length is the tactical video: 199,319 frames at 25 fps.
N_FRAMES = 199319
FPS = 25
PITCH = (105.3, 68.0)  # metres, length x width

# Raw ATD line: "<frame>:<player>;<player>;...:<ball>", each group "x,y,speed".
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"  # first character of "NaN"

HOME, AWAY, NOBODY = 0, 1, -1

# Club names as they appear in the Automatic Tracking Data and Smart Tagging.
HOME_NAMES = frozenset({"Barcelona"})
AWAY_NAMES = frozenset({"Atletic Club"})

# Phase codes used as possession priors. Other codes are ignored.
STRONG_CODES = frozenset({"BUILD UP", "CREATING CHANCES", "LONG BALL"})
STRONG_DEF_CODES = frozenset({
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN ATTACKING THIRD",
})

# Sticky assignment and dead-ball seeds.
CLOSE_M = 6.0
SWITCH_NEED = 8
GAP_FILL = 25
GAP_KILL = 75
MISS_D_RUN = 30
SET_LEAD = 50
FALLBACK_PREFIX = 100
SMOOTH_W = 125  # half-width; the interior window has 251 frames
OUT_MARGIN_M = 0.5

# Restart-kick detection for SET PIECES dead windows.
REF_SEARCH = 100
KICK_DISP_M = 1.0
KICK_CONSEC = 3
MIN_PREFIX = 10
MAX_PREFIX = 250

# Player activity and point-cloud features.
SPEED_LO = 0.0
SPEED_HI = 12.0
MOVE_MPS = 1.5
RUN_MPS = 3.0
SLOW_MPS = 1.2
NEAR_BOUND_M = 5.0
BALL_CLUSTER_M = 8.0
MIN_POS_SPREAD = 6
MIN_TEAM_SPREAD = 4

# Activity overlay written after smoothing.
C0_MED_LOW = 1.25
C0_SLOW_FRAC = 0.55
C0_MIN_VALID = 8
C0_MIN_TEAM = 5
C0_SUSTAIN = 25

# Live-evidence alignment check.
ALIGN_MIN_SPEED_MPS = 0.8
ALIGN_MAX_ANGLE_DEG = 45.0
GEO_LIVE_DIST_M = 2.0

# Temporal correction applied after the activity overlay.
RELEASE_FALSE_D = True
GEO_STREAK = 12
MED_THR = 1.8
MOVE_FRAC = 0.45
MIN_HA_FRAMES = 30
BRIDGE_GAP_FRAMES = 30
BRIDGE_ACROSS = "D_only"
PROTECT_D_MIN = 75

# Official DATA.md filenames. The tactical source is the submission index.
SOURCE = "tactical"
STEM = "BAR-ATH_tactical"
ASSET_FILES = {
    "atd_meta": "BAR-ATH_tactical_atd.xml",
    "atd_raw": "BAR-ATH_tactical_atd.txt",
    "smart_tagging": "BAR-ATH_tactical_smart_tagging.xml",
}

EXPECTED_SHA256 = "d2f7649ad9ee44924e794e88d29bad4114fca06c3aa7b0b1d3dc1702247d2232"

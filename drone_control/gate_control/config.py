"""Config for the recurrent vision gate-policy deployment (gate_flight.py).

DRY_RUN computes + prints but never arms; flip it off (and DRY-RUN-verify the AETR
signs) before flying. Bounds are launch-relative — the policy is gate-relative, so
nothing here references the gate's location. Tune for your room.
"""
import os

DRY_RUN = True

GATE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "gate")

# Automated kills (cut + disarm + exit), plus manual TX12 disarm and SPACEBAR.
MAX_RADIUS_M = 4.0                 # horizontal distance from launch
MAX_ALT_ABOVE_LAUNCH_M = 2.5       # climb above launch (gate is ~1.5 m up)
MAX_DESCENT_BELOW_LAUNCH_M = 0.5   # sink below launch
VICON_KILL_S = 0.60                # pose lost this long

# Battery cutoff (CRSF telemetry).
LOW_BATT_CUTOFF = True
BATT_PRESENT_V = 2.5               # below this = no valid pack reading, ignore
MIN_CELL_V = 3.3
CELLS = 1

GATE_MASK_STALE_S = 0.30           # status-line warning only, not a kill
GATE_MASK_CONF = None              # threshold the GateNet prob, or None for soft

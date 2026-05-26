"""Generate a printable checkerboard for camera calibration.

Run:    .venv/bin/python camera_setup/make_checkerboard.py
Print:  open camera_setup/checkerboard.png, print at 100% / 'Actual Size'
        (NOT fit-to-page).
Then:   measure ONE black square's edge with a ruler in mm. Convert to meters
        and put the value into SQUARE_SIZE_M in calibrate_camera.py.
        e.g. 21mm -> SQUARE_SIZE_M = 0.021
"""
import os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "checkerboard.png")

COLS, ROWS = 10, 7  # squares -> 9x6 inner corners
PX_PER_SQUARE = 200
MARGIN = 100

w = COLS * PX_PER_SQUARE + 2 * MARGIN
h = ROWS * PX_PER_SQUARE + 2 * MARGIN
img = np.full((h, w), 255, dtype=np.uint8)
for r in range(ROWS):
    for c in range(COLS):
        if (r + c) % 2 == 0:
            y1, y2 = MARGIN + r * PX_PER_SQUARE, MARGIN + (r + 1) * PX_PER_SQUARE
            x1, x2 = MARGIN + c * PX_PER_SQUARE, MARGIN + (c + 1) * PX_PER_SQUARE
            img[y1:y2, x1:x2] = 0

cv2.imwrite(OUTPUT, img)
print(f"wrote {OUTPUT} ({w}x{h} px, {COLS-1}x{ROWS-1} inner corners)")
print("print at 100% (Actual Size, no scaling), then measure ONE square edge with a ruler.")

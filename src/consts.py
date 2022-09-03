import numpy as np

DEFAULT_HEIGHT = 480

HAAR_TEMPLATES = [
    np.array([[0.0, 0.0, 1.0, 0.5]]),  # left-right; every row represents white rectangle (j, k, h, w) within template
    np.array([[0.0, 0.0, 0.5, 1.0]]),  # top-down
    np.array([[0.0, 0.25, 1.0, 0.5]]),  # left-center-right
    np.array([[0.25, 0.0, 0.5, 1.0]]),  # top-center-down
    np.array([[0.0, 0.0, 0.5, 0.5],  # diagonal
              [0.5, 0.5, 0.5, 0.5]])
]
FEATURE_MIN = 0.25
FEATURE_MAX = 0.5

DETECTION_SCALES = 4
DETECTION_W_GROWTH = 1.2
DETECTION_W_JUMP_RATIO = 0.1

DETECTION_SIZES = [
    [45, 15],
    # [75, 25],
    # [105, 35],
    # [165, 55],
]

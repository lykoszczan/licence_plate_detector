import cv2
import numpy as np
from numba import jit


def haar_features(ii, j0, k0, hcws_subset, n, feature_indexes=None, verbose=False, i_scaled=None):
    # verbose = True

    features = np.zeros(n, dtype="int16")
    if feature_indexes is None:
        feature_indexes = list(range(n))
    for i, fi in enumerate(feature_indexes):
        features[fi] = haar_feature(ii, j0, k0, hcws_subset[i])

        if verbose:
            cv2.imshow("DEMO", draw_haar_feature_at(i_scaled, j0, k0, hcws_subset[i]))
            cv2.waitKey()
    return features


@jit(nopython=True, cache=True)
def haar_feature(ii, j0, k0, hcw):
    j, k, h, w = hcw[0]
    j1 = j0 + j
    k1 = k0 + k
    total_intensity = ii_delta(ii, j1, k1, j1 + h - 1, k1 + w - 1)
    total_area = h * w
    white_intensity = 0
    white_area = 0
    for white in hcw[1:]:
        j, k, h, w = white
        j1 = j0 + j
        k1 = k0 + k
        white_intensity += ii_delta(ii, j1, k1, j1 + h - 1, k1 + w - 1)
        white_area += h * w
    black_intensity = total_intensity - white_intensity
    black_area = total_area - white_area
    return np.int16(white_intensity / white_area - black_intensity / black_area)


@jit(nopython=True, cache=True)
def ii_delta(ii, j1, k1, j2, k2):
    delta = ii[j2, k2]
    if j1 > 0:
        delta -= ii[j1 - 1, k2]
    if k1 > 0:
        delta -= ii[j2, k1 - 1]
    if j1 > 0 and k1 > 0:
        delta += ii[j1 - 1, k1 - 1]
    return delta


def draw_haar_feature_at(i, j0, k0, hcw):
    i_copy = i.copy()
    j, k, h, w = hcw[0]
    j1 = j0 + j
    k1 = k0 + k
    cv2.rectangle(i_copy, (k1, j1), (k1 + w - 1, j1 + h - 1), (0, 0, 0), cv2.FILLED)
    for white in hcw[1:]:
        j, k, h, w = white
        j1 = j0 + j
        k1 = k0 + k
        cv2.rectangle(i_copy, (k1, j1), (k1 + w - 1, j1 + h - 1), (255, 255, 255), cv2.FILLED)
    return cv2.addWeighted(i, 0.4, i_copy, 0.6, 0.0)

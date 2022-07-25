import numpy as np

import consts
import utils
from haar_features import haar_features


def detection_one_scale(H, W, h, w, threshold, detections, clf, feature_indexes, n, hcs, ii):
    dj = int(np.round(w * consts.DETECTION_W_JUMP_RATIO))
    dk = dj
    print(f"S: {w}x{h}, DJ: {dj}, DK: {dk}")
    rj = int(((H - h) % dj) / 2)
    rk = int(((W - w) % dk) / 2)
    hcws = utils.multiplyWindow(w, h, hcs)
    hcws = [hcw.astype("int32") for hcw in hcws]
    for j in range(rj, H - h, dj):
        for k in range(rk, W - w, dk):
            features = haar_features(ii, j, k, hcws, n, feature_indexes)

            decision = clf.decision_function(np.array([features]))
            if decision > threshold:
                detections.append([j, k, h, w])
                print(f"! FACE DETECTED, DECISION: {decision}, size: {w}x{h}")

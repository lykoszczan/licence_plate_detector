from functools import reduce

import numpy as np
from joblib import Parallel, delayed

import consts
import utils
from haar_features import haar_features


def detection_one_scale(H, W, h, w, threshold, detections, clf, feature_indexes, n, hcs, ii, orgImg=None):
    dj = int(np.round(w * consts.DETECTION_W_JUMP_RATIO))
    dk = int(np.round(h * consts.DETECTION_W_JUMP_RATIO))
    print(f"S: {w}x{h}, DJ: {dj}, DK: {dk}")
    rj = int(((H - h) % dj) / 2)
    rk = int(((W - w) % dk) / 2)
    hcws = utils.multiplyWindow(w, h, hcs)
    hcws = [hcw.astype("int32") for hcw in hcws]

    counter = 0
    for j in range(rj, H - h, dj):
        for k in range(rk, W - w, dk):
            counter += 1
    print('all windows count ' + str(counter))

    detections = []
    with Parallel(n_jobs=16, verbose=0) as parallel:
        def the_job(j0):
            job_detections = []
            for k0 in range(rk, W - w + 1, dk):
                features = haar_features(ii, j0, k0, hcws, n, feature_indexes)
                decision = clf.decision_function(np.array([features]))
                if decision > threshold:
                    job_detections.append([j0, k0, h, w])
                    print(f"! PLATE DETECTED, DECISION: {decision}, size: {w}x{h}")
            return job_detections

        job_detections_all = parallel((delayed(the_job)(j0) for j0 in range(rj, H - h + 1, dj)))
        if job_detections_all:
            detections.extend(reduce(lambda a, b: a + b, [jb for jb in job_detections_all]))

    return detections

    # for j in range(rj, H - h, dj):
    #     # joblib
    #     for k in range(rk, W - w, dk):
    #         features = haar_features(ii, j, k, hcws, n, feature_indexes)
    #         # if not orgImg is None:
    #         #     for hcw in hcws:
    #         #         p1 = (k, j)
    #         #         p2 = (k + w - 1, j + h - 1)
    #         #         # hcw = np.array([[1, 7, 3, 22], [1, 7, 1, 22]], dtype=object)
    #         #         cv2.rectangle(orgImg, p1, p2, (0, 0, 255), 1)
    #         #         cv2.imshow("DEMO", draw_haar_feature_at(orgImg, j, k, hcw))
    #         #         cv2.waitKey()
    #         decision = clf.decision_function(np.array([features]))
    #         if decision > threshold:
    #             detections.append([j, k, h, w])
    #             print(f"! PLATE DETECTED, DECISION: {decision}, size: {w}x{h}")

import numpy as np
import cv2
import consts


def haar_indexes(s, p):
    indexes = []
    for t in range(len(consts.HAAR_TEMPLATES)):
        for s_j in range(s):
            for s_k in range(s):
                for p_j in range(-(p - 1), p):
                    for p_k in range(-(p - 1), p):
                        indexes.append(np.array([t, s_j, s_k, p_j, p_k]))

    return np.array(indexes)

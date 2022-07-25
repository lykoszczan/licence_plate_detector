import numpy as np
import cv2
import utils
import core
import time
from numba import jit
import pickle
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt


def getWindowParameters():
    w = 150
    h = 40
    j0 = 380
    k0 = 200

    return [w, h, j0, k0]


def getHaarParameters():
    s = 3  # 5
    p = 4  # 5

    return [s, p]


if __name__ == "__main__":
    i = cv2.imread('test_data/car.png')

    i_scaled = utils.scale_image(i)
    i_gray = cv2.cvtColor(i_scaled, cv2.COLOR_BGR2GRAY)
    [w, h, j0, k0] = getWindowParameters()
    cv2.rectangle(i_scaled, (k0, j0), (k0 + w - 1, j0 + h - 1), (0, 0, 255), 1)
    cv2.imshow("DEMO", i_scaled)
    cv2.waitKey()

    [s, p] = getHaarParameters()
    indexes = core.haar_indexes(s, p)
    n = indexes.shape[0]
    print("N: " + str(n))

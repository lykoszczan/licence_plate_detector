import time
import pickle
import numpy as np
import cv2
import consts

def multiplyWindow(w, h, hcws):
    tmp = list(range(len(hcws)))
    for i in range(0, len(hcws)):
        tmp[i] = hcws[i] * [h, w, h, w]

    return tmp

def scale_image(i):
    h, w, _ = i.shape
    w_new = int(np.round(w * consts.DEFAULT_HEIGHT / h))
    return cv2.resize(i, (w_new, consts.DEFAULT_HEIGHT))


def readDataFile():
    outputPath = r'C:\Users\lykos\Desktop\py-mgr\src\tools\data.txt'

    with open(outputPath, 'r') as f:
        lines = f.readlines()

    return lines


def readImage(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(path)
        exit(500)

    return img


def pickle_all(fname, some_list):
    print("PICKLE...")
    t1 = time.time()
    f = open(fname, "wb+")
    pickle.dump(some_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    t2 = time.time()
    print("PICKLE DONE. [TIME: " + str(t2 - t1) + " s.]")


def unpickle_all(fname):
    print("UNPICKLE...", fname)
    t1 = time.time()
    f = open(fname, "rb")
    some_list = pickle.load(f)
    f.close()
    t2 = time.time()
    print("UNPICKLE DONE. [TIME: " + str(t2 - t1) + " s.]")
    return some_list

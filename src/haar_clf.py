import numpy as np
import cv2
import time
from numba import jit
import copy
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from objects.ParsedLine import ParsedLine
import utils
import training
import consts
import random

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

DETECTION_SCALES = 1
DETECTION_W_MIN = 140
DETECTION_W_GROWTH = 1.2
DETECTION_W_JUMP_RATIO = 0.1


def is_intersect(rec1, rec2):
    if (rec1[0][0] < rec2[0][0]) or (rec1[0][1] < rec2[0][1]):
        if (rec1[1][0] > rec2[0][0]) or (rec1[1][1] > rec2[0][1]):
            return True
    elif (rec1[0][0] > rec2[0][0]) or (rec1[0][1] > rec2[0][1]):
        if (rec1[0][0] < rec2[1][0]) or (rec1[0][1] < rec2[1][1]):
            return True
    return False


def overlapping_ratio(rec1, rec2):
    rec1_area = (rec1[1][0] - rec1[0][0]) * (rec1[1][1] - rec1[0][1])
    rec2_area = (rec1[1][0] - rec1[0][0]) * (rec1[1][1] - rec1[0][1])

    xx = max(rec1[0][0], rec2[0][0])
    yy = max(rec1[0][1], rec2[0][1])
    aa = min(rec1[1][0], rec2[1][0])
    bb = min(rec1[1][1], rec2[1][1])

    width = max(0, aa - xx)
    height = max(0, bb - yy)

    intersection_area = width * height

    union_area = rec1_area + rec2_area - intersection_area

    return intersection_area / union_area


def calculate_new_rectangle(rec1, rec2, ratio):
    new_rec = []
    if ratio > 0.3:
        new_rec.append((int((rec1[0][0] + rec2[0][0]) / 2), int((rec1[0][1] + rec2[0][1]) / 2)))
        new_rec.append((int((rec1[1][0] + rec2[1][0]) / 2), int((rec1[1][1] + rec2[1][1]) / 2)))
    else:
        xx = min(rec1[0][0], rec2[0][0])
        yy = min(rec1[0][1], rec2[0][1])
        aa = max(rec1[1][0], rec2[1][0])
        bb = max(rec1[1][1], rec2[1][1])
        new_rec.append((xx, yy))
        new_rec.append((aa, bb))
    return new_rec


def non_max_supression(detections, ratio):
    rects = []
    for j, k, h, w in detections:
        rects.append([(k, j), (k + w - 1, j + h - 1)])
    run = True
    while run:
        run = False
        for i in range(0, len(rects)):
            for j in range(0, len(rects)):
                if i == j:
                    continue
                if is_intersect(rects[i], rects[j]) and overlapping_ratio(rects[i], rects[j]) > ratio:
                    rec1 = rects.pop(i)
                    if i < j:
                        rec2 = rects.pop(j - 1)
                    else:
                        rec2 = rects.pop(j)
                    rects.append(calculate_new_rectangle(rec1, rec2, overlapping_ratio(rec1, rec2)))
                    run = True
                    break
            if run:
                break
    return rects


def haar_indexes(s, p):
    indexes = []
    for t in range(len(HAAR_TEMPLATES)):
        for s_j in range(s):
            for s_k in range(s):
                for p_j in range(-(p - 1), p):
                    for p_k in range(-(p - 1), p):
                        indexes.append(np.array([t, s_j, s_k, p_j, p_k]))
    return np.array(indexes)


def haar_coords(s, p, indexes):
    coords = []
    f_jump = (FEATURE_MAX - FEATURE_MIN) / (s - 1)
    for t, s_j, s_k, p_j, p_k in indexes:
        f_h = FEATURE_MIN + s_j * f_jump
        f_w = FEATURE_MIN + s_k * f_jump
        p_jump_h = (1.0 - f_h) / (2 * p - 2)
        p_jump_w = (1.0 - f_w) / (2 * p - 2)
        pos_j = 0.5 + p_j * p_jump_h - 0.5 * f_h
        pos_k = 0.5 + p_k * p_jump_w - 0.5 * f_w
        single_coords = [np.array([pos_j, pos_k, f_h, f_w])]  # whole rectangle for single feature
        for white in HAAR_TEMPLATES[t]:
            white_coords = np.array([pos_j, pos_k, 0.0, 0.0]) + white * np.array([f_h, f_w, f_h, f_w])
            single_coords.append(white_coords)
        coords.append(np.array(single_coords))
    return np.array(coords, dtype=object)


def integral_image(i):
    ii = i.cumsum(axis=0).cumsum(axis=1).astype("int32")
    return ii


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


def haar_features(ii, j0, k0, hcws_subset, n, feature_indexes=None, verbose=False):
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


# intersection over union - czesc wspolna
def iou(coords_1, coords_2):
    j11, k11, j12, k12 = coords_1
    j21, k21, j22, k22 = coords_2
    dj = np.min([j12, j22]) - np.max([j21, j11]) + 1
    if dj <= 0:
        return 0.0
    dk = np.min([k12, k22]) - np.max([k21, k11]) + 1
    if dk <= 0:
        return 0.0
    i = dj * dk
    u = (j12 - j11 + 1) * (k12 - k11 + 1) + (j22 - j21 + 1) * (k22 - k21 + 1) - i
    return i / u


def fddb_read_single_fold(n_negs_per_img, hfs_coords, n, parsedObject, verbose=False, fold_title=""):
    np.random.seed(1)

    verbose = False
    # verbose = parsedObject.elementsCount > 1
    showFeatures = False

    # settings for sampling negatives
    w_relative_min = 0.01
    w_relative_max = 0.10
    w_relative_spread = w_relative_max - w_relative_min
    neg_max_iou = 0.5

    X_list = []
    y_list = []

    file_name = parsedObject.path

    n_img = 0
    n_negative_probes = 0
    counter = 0
    while True:
        log_line = str(counter) + ": [" + file_name + "]"
        if fold_title != "":
            log_line += " [" + fold_title + "]"
        print(log_line)
        counter += 1

        i0 = utils.readImage(parsedObject.path)
        i = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY)
        ii = integral_image(i)

        n_img_faces = parsedObject.elementsCount
        img_faces_coords = []
        for z in range(n_img_faces):
            (xmin, ymin) = parsedObject.rects[z][0]
            (xmax, ymax) = parsedObject.rects[z][1]
            w = xmax - xmin
            h = ymax - ymin
            j0 = ymin
            k0 = xmin

            img_face_coords = np.array([xmin, ymin, xmax, ymax])
            if j0 < 0 or k0 < 0 or j0 + h - 1 >= i.shape[0] or k0 + w - 1 >= i.shape[1]:
                if verbose:
                    print("WINDOW " + str(img_face_coords) + " OUT OF BOUNDS. [IGNORED]")
                continue

            if verbose:
                print('factor w', w / ii.shape[0])
                print('factor y', h / ii.shape[1])

            # min relative size of positive window (smaller may lead to division by zero when white regions in haar features have no area)
            if (w / ii.shape[0] < 0.01):
                print("WINDOW " + str(img_face_coords) + " TOO SMALL. [IGNORED]")
                exit(500)
            img_faces_coords.append(img_face_coords)
            if verbose:
                p1 = (k0, j0)
                p2 = (k0 + w - 1, j0 + h - 1)
                cv2.rectangle(i0, p1, p2, (0, 0, 255), 1)
                # cv2.imshow("FDDB", i0)
                cv2.waitKey()
            hfs_coords_window = multiplyWindow(w, h, hfs_coords)
            hfs_coords_window = np.array(list(map(lambda npa: npa.astype("int32"), hfs_coords_window)), dtype=object)
            feats = haar_features(ii, j0, k0, hfs_coords_window, n)

            # wyswietlenie cech
            if showFeatures:
                for hcw in hfs_coords_window:
                    cv2.imshow("DEMO", draw_haar_feature_at(i0, j0, k0, hcw))
                    cv2.waitKey()
            if verbose:
                print("POSITIVE WINDOW " + str(img_face_coords) + " ACCEPTED. FEATURES: " + str(feats) + ".")
                cv2.waitKey(0)
            n_img += 1
            X_list.append(feats)
            y_list.append(1)
        for z in range(n_negs_per_img * parsedObject.elementsCount):
            while True:
                # wymiary tablict rejestracyjnej to 520x114 czyli szerokosc musi byc ok 4.5 raza wieksza
                w_random = int((random.random() * w_relative_spread + w_relative_min) * i.shape[1])
                h_random = int(np.round(w_random / random.randrange(3, 5, 1)))

                k0 = random.randint(150, i.shape[1] - 150)  # szer
                j0 = random.randint(150, i.shape[0] - 250)  # wys

                if verbose:
                    # area for negative windows beginnings
                    cv2.rectangle(i0, (150, 150), (i.shape[1] - 150, i.shape[0] - 250), (255, 255, 255), 1)

                patch = np.array([k0, j0, k0 + w_random - 1, j0 + h_random - 1])
                ious = list(map(lambda ifc: iou(patch, ifc), img_faces_coords))
                max_iou = max(ious) if len(ious) > 0 else 0.0
                if max_iou < neg_max_iou:
                    hfs_coords_window = multiplyWindow(w_random, h_random, hfs_coords)
                    hfs_coords_window = np.array(list(map(lambda npa: npa.astype("int32"), hfs_coords_window)),
                                                 dtype=object)
                    feats = haar_features(ii, j0, k0, hfs_coords_window, n)
                    n_negative_probes += 1
                    X_list.append(feats)
                    y_list.append(-1)
                    if verbose:
                        print("NEGATIVE WINDOW " + str(patch) + " ACCEPTED. FEATURES: " + str(feats) + ".")
                        p1 = (k0, j0)
                        p2 = (k0 + w_random - 1, j0 + h_random - 1)
                        cv2.rectangle(i0, p1, p2, (0, 255, 0), 1)
                        # cv2.imshow("FDDB", i0)
                        cv2.waitKey(0)
                    break
                else:
                    if verbose:
                        print("NEGATIVE WINDOW " + str(patch) + " IGNORED. [MAX IOU: " + str(max_iou) + "]")
                        p1 = (k0, j0)
                        p2 = (k0 + w_random - 1, j0 + h_random - 1)
                        cv2.rectangle(i0, p1, p2, (255, 255, 0), 1)
                        cv2.waitKey()
        if verbose:
            cv2.imshow("FDDB", i0)
            cv2.waitKey(0)
        break
    print("IMAGES IN THIS FOLD: " + str(parsedObject.elementsCount) + ".")
    print("ACCEPTED PROBES IN THIS FOLD: " + str(n_img) + ".")
    print("NEGATIVE PROBES IN THIS FOLD: " + str(n_negative_probes) + ".")

    X = np.stack(X_list)
    y = np.stack(y_list)
    return X, y


def fddb_data(path_fddb_root, hfs_coords, n_negs_per_img, n):
    n_negs_per_img = n_negs_per_img

    fold_paths_all = utils.readDataFile()
    fold_paths_train = fold_paths_all[0:10]
    fold_paths_test = fold_paths_all[-2:]
    X_train = None
    y_train = None
    for index, fold_path in enumerate(fold_paths_train):
        obj = ParsedLine(fold_path)
        print("PROCESSING TRAIN FOLD " + str(index + 1) + "/" + str(len(fold_paths_train)) + "...")
        t1 = time.time()
        X, y = fddb_read_single_fold(n_negs_per_img, hfs_coords, n, fold_title=obj.path, parsedObject=obj)
        t2 = time.time()
        print("PROCESSING TRAIN FOLD " + str(index + 1) + "/" + str(len(fold_paths_train)) + " DONE IN " + str(
            t2 - t1) + " s.")
        print("---")
        if X_train is None:
            X_train = X
            y_train = y
        else:
            X_train = np.r_[X_train, X]
            y_train = np.r_[y_train, y]

    X_test = None
    y_test = None
    for index, fold_path in enumerate(fold_paths_test):
        obj = ParsedLine(fold_path)
        print("PROCESSING TEST FOLD " + str(index + 1) + "/" + str(len(fold_paths_test)) + "...")
        t1 = time.time()
        X, y = fddb_read_single_fold(n_negs_per_img, hfs_coords, n, fold_title=fold_path, parsedObject=obj)
        t2 = time.time()
        print("PROCESSING TEST FOLD " + str(index + 1) + "/" + str(len(fold_paths_test)) + " DONE IN " + str(
            t2 - t1) + " s.")
        print("---")
        if X_test is None:
            X_test = X
            y_test = y
        else:
            X_test = np.r_[X_test, X]
            y_test = np.r_[y_test, y]
    print("TRAIN DATA SHAPE: " + str(X_train.shape))
    print("TEST DATA SHAPE: " + str(X_test.shape))
    return X_train, y_train, X_test, y_test


def detect(i_scaled, ii, clf, hcs, feature_indexes, threshold=0.0):
    H, W = ii.shape
    detectionTime = 0
    calcFeatureTime = 0
    windows_count = 0
    for s in consts.DETECTION_SIZES:
        [w, h] = s
        dj = int(np.round(w * DETECTION_W_JUMP_RATIO))
        dk = dj
        rj = int(((H - h) % dj) / 2)
        rk = int(((W - w) % dk) / 2)
        for j in range(rj, H - h, dj):
            for k in range(rk, W - w, dk):
                windows_count += 1

    n = hcs.size
    # chyba po to aby liczyc tylko wybrane indeksy
    hcs = hcs[feature_indexes]
    detections = []
    window_index = 0
    progress_print = int(0.01 * windows_count)
    print("DETECTION...")
    t1 = time.time()
    for s in consts.DETECTION_SIZES:
        [w, h] = s

        dj = int(np.round(w * DETECTION_W_JUMP_RATIO))
        dk = dj
        print(f"S: {w}x{h}, DJ: {dj}, DK: {dk}")
        rj = int(((H - h) % dj) / 2)
        rk = int(((W - w) % dk) / 2)
        hcws = multiplyWindow(w, h, hcs)
        hcws = [hcw.astype("int32") for hcw in hcws]
        for j in range(rj, H - h, dj):
            for k in range(rk, W - w, dk):
                tCalc = time.time()
                features = haar_features(ii, j, k, hcws, n, feature_indexes)
                calcFeatureTime += (time.time() - tCalc)

                tDec = time.time()
                decision = clf.decision_function(np.array([features]))
                detectionTime += (time.time() - tDec)
                if decision > threshold:
                    detections.append([j, k, h, w])
                    print(f"! FACE DETECTED, DECISION: {decision}, size: {w}x{h}")
                window_index += 1
                if (window_index % progress_print == 0):
                    print(f"PROGRESS: {window_index / windows_count}")
    t2 = time.time()
    print(f"Decision function time  {detectionTime} s")
    print(f"Features time {calcFeatureTime} s")
    print(f"DETECTION DONE IN {t2 - t1} s")


    # bez łączenia
    for j, k, h, w in detections:
        cv2.rectangle(i_scaled, (k, j), (k + w - 1, j + h - 1), (0, 0, 255), 1)
    cv2.imshow("OUTPUT_all", i_scaled)
    cv2.waitKey()

    # połaczone
    # rects = non_max_supression(detections, 0.1)
    # for rect in rects:
    #     cv2.rectangle(i_scaled, rect[0], rect[1], (0, 0, 255), 1)
    # cv2.imshow("OUTPUT", i_scaled)
    # cv2.waitKey()


def generateROC(clf):
    y_score = clf.decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(
        fpr[0],
        tpr[0],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[0],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()


def multiplyWindow(w, h, hcws):
    return multiplyWindowCalc(w, h, hcws)


# @jit(nopython=True)
def multiplyWindowCalc(w, h, hcws):
    tmp = list(range(len(hcws)))
    for i in range(0, len(hcws)):
        tmp[i] = hcws[i] * [h, w, h, w]

    return tmp


clf_path = "clf/"
data_path = "trained/"

s = 3
p = 4

indexes = haar_indexes(s, p)
n = indexes.shape[0]  # number of all features
print("N: " + str(n))
hcs = haar_coords(s, p, indexes)

data_name = "licence_plates_n_" + str(n) + "_s_" + str(s) + "_p_" + str(p) + ".bin"
# X_train, y_train, X_test, y_test = fddb_data("annotations", hcs, 50, n)
# utils.pickle_all(data_path + data_name, [X_train, y_train, X_test, y_test])
X_train, y_train, X_test, y_test = utils.unpickle_all(data_path + data_name)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_train.dtype)

clf = training.learn(X_train=X_train, y_train=y_train, s=s, n=n, p=p, clf_path=clf_path, adaBoost=False)

indexes_pos = y_test == 1
indexes_neg = y_test == -1
print(f"ACC TEST: {clf.score(X_test, y_test)}")
print(f"SENSITIVITY TEST: {clf.score(X_test[indexes_pos], y_test[indexes_pos])}")
print(f"SPECIFITY TEST: {clf.score(X_test[indexes_neg], y_test[indexes_neg])}")

generateROC(clf)

# feature_indexes = clf.feature_importances_ > 0  # Ada
feature_indexes = clf.feature_indexes_

# i = cv2.imread("test_data/car.png")
i = cv2.imread("test_data/camera.jpg")

i_scaled = utils.scale_image(i)
i_gray = cv2.cvtColor(i_scaled, cv2.COLOR_BGR2GRAY)
# remove subtitles from camera
i_gray_cropped = i_gray[0:-80, 0:]
# cv2.imshow("cropped", i_gray_cropped)
# cv2.waitKey()
ii = integral_image(i_gray_cropped)

# w = 150
# h = 40
# j0 = 380
# k0 = 200
# hcws = multiplyWindow(w, h, hcs)
# hcws = [hcw.astype("int32") for hcw in hcws]
#
# for hcw in hcws:
#     print("HCW: ")
#     print(hcw)
#     print("VALUE: " + str(haar_feature(ii, j0, k0, hcw)))
#     cv2.imshow("DEMO", draw_haar_feature_at(i_scaled, j0, k0, hcw))
#     cv2.waitKey()


detect(i_scaled, ii, clf, hcs, feature_indexes, threshold=1.6)

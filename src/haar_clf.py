import cv2
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import random
import time
from sklearn.metrics import roc_curve, auc

import consts
import training
import utils
from detection import detection_one_scale
from haar_features import draw_haar_feature_at, haar_features
from objects.ParsedLine import ParsedLine
from src.ocr_detection import detect_licence_plate_characters


def test_video(path):
    cap = cv2.VideoCapture(path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        count = count + 1

        if (count % 30 != 0):
            continue

        i_scaled = utils.scale_image(frame)
        i_gray = cv2.cvtColor(i_scaled, cv2.COLOR_BGR2GRAY)
        i_gray_cropped = i_gray[0:-80, 0:]
        ii = integral_image(i_gray_cropped)
        i_scaled = detect(i_scaled, ii, clf, hcs, feature_indexes, threshold=2.1, original_image=frame,
                          show_output=False, ocr=True)

        cv2.imshow('window-name', i_scaled)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # destroy all opened windows


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
            hfs_coords_window = utils.multiplyWindow(w, h, hfs_coords)
            hfs_coords_window = np.array(list(map(lambda npa: npa.astype("int32"), hfs_coords_window)), dtype=object)
            try:
                feats = haar_features(ii, j0, k0, hfs_coords_window, n)
            except Exception as e:
                print(e)
                print('factor w', w / ii.shape[0])
                print('factor y', h / ii.shape[1])
                print("haar feature error")
                continue

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

                # k0 = random.randint(150, i.shape[1] - 150)  # szer
                # j0 = random.randint(150, i.shape[0] - 250)  # wys

                k0 = random.randint(0, i.shape[1] - w_random - 1)  # szer
                j0 = random.randint(0, i.shape[0] - h_random - 1)  # wys

                if verbose:
                    print('factor w', w_random / i.shape[0])
                    print('factor y', h_random / i.shape[1])

                if h_random / ii.shape[1] < 0.004:
                    continue

                if verbose:
                    # area for negative windows beginnings
                    cv2.rectangle(i0, (150, 150), (i.shape[1] - 150, i.shape[0] - 250), (255, 255, 255), 1)

                patch = np.array([k0, j0, k0 + w_random - 1, j0 + h_random - 1])
                ious = list(map(lambda ifc: iou(patch, ifc), img_faces_coords))
                max_iou = max(ious) if len(ious) > 0 else 0.0
                if max_iou < neg_max_iou:
                    hfs_coords_window = utils.multiplyWindow(w_random, h_random, hfs_coords)
                    hfs_coords_window = np.array(list(map(lambda npa: npa.astype("int32"), hfs_coords_window)),
                                                 dtype=object)

                    try:
                        feats = haar_features(ii, j0, k0, hfs_coords_window, n)
                    except Exception as e:
                        print(e)
                        print('factor w', w_random / ii.shape[0])
                        print('factor y', h_random / ii.shape[1])
                        print("haar feature negative probe error")
                        continue

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


def fddb_data(hfs_coords, n_negs_per_img, n):
    n_negs_per_img = n_negs_per_img

    fold_paths_all = utils.readDataFile()
    fold_paths_train = fold_paths_all[0:-100]
    fold_paths_test = fold_paths_all[-100:]
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


def detect(i_scaled, ii, clf, hcs, feature_indexes, threshold=0.0, original_image=None, show_output=False, ocr=False):
    H, W = ii.shape
    n = hcs.size
    # chyba po to aby liczyc tylko wybrane indeksy
    hcs = hcs[feature_indexes]

    manager = mp.Manager()
    detections = manager.list()

    print("DETECTION...")
    t1 = time.time()
    procs = []

    for s in consts.DETECTION_SIZES:
        [w, h] = s
        p = mp.Process(target=detection_one_scale,
                       args=(H, W, h, w, threshold, detections, clf, feature_indexes, n, hcs, ii))
        p.start()
        procs.append(p)

    for p in procs: p.join()

    # bez łączenia
    # for j, k, h, w in detections:
    #     cv2.rectangle(i_scaled, (k, j), (k + w - 1, j + h - 1), (0, 0, 255), 1)
    # cv2.imshow("OUTPUT_all", i_scaled)
    # cv2.waitKey()

    # połaczone
    rects = non_max_supression(detections, 0.1)
    for rect in rects:
        cv2.rectangle(i_scaled, rect[0], rect[1], (0, 0, 255), 1)

        [k, j] = rect[0]
        [k_end, j_end] = rect[1]
        h = j_end - j
        w = k_end - k
        [j0, k0, h0, w0] = utils.transform_to_original(j, k, h, w, i_scaled, original_image)
        rect_cropped = original_image[j0:j0 + h0 - 1, k0:k0 + w0 - 1]
        # cv2.imshow("OUTPUT", rect_cropped)
        # cv2.waitKey()

        if ocr:
            plate_text = detect_licence_plate_characters(rect_cropped)
            if plate_text:
                print(plate_text)
                if len(plate_text) > 3:
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(i_scaled, plate_text, (k + 2, j - 5), font, 0.5, (0, 0, 255), 1)

    t2 = time.time()
    print(f"DETECTION DONE IN {t2 - t1} s")

    if show_output:
        cv2.imshow("OUTPUT", i_scaled)
        cv2.waitKey()

    return i_scaled


def generate_roc(clf):
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


clf_path = "clf/"
data_path = "trained/"

s = 3
p = 4

indexes = haar_indexes(s, p)
n = indexes.shape[0]  # number of all features
print("N: " + str(n))
hcs = haar_coords(s, p, indexes)

neg_per_image = 50
data_name = "licence_plates_n_" + str(n) + "_s_" + str(s) + "_p_" + str(p) + "_negs_" + str(neg_per_image) + ".bin"
# X_train, y_train, X_test, y_test = fddb_data(hcs, neg_per_image, n)
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

# generate_roc(clf)

# feature_indexes = clf.feature_importances_ > 0  # Ada
feature_indexes = clf.feature_indexes_

test_video("test_data/video/dr750x-plus-dzien.mp4")
exit()

# i = cv2.imread("test_data/car.png")
i = cv2.imread("test_data/camera.jpg")

i_scaled = utils.scale_image(i)
i_gray = cv2.cvtColor(i_scaled, cv2.COLOR_BGR2GRAY)
# remove subtitles from camera
i_gray_cropped = i_gray[0:-80, 0:]
# cv2.imshow("cropped", i_gray_cropped)
# cv2.waitKey()
ii = integral_image(i_gray_cropped)

detect(i_scaled, ii, clf, hcs, feature_indexes, threshold=1.6, original_image=i, show_output=True, ocr=True)

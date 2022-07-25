import time
from sklearn.ensemble import AdaBoostClassifier
import utils
import boosting


def learn(X_train, y_train, clf_path, n, s, p, adaBoost=False, force=False):
    T = 64
    B = 8
    if adaBoost:
        clf_name = "clf_sklearn_rb_plates_n_" + str(n) + "_s_" + str(s) + "_p_" + str(p) + "_T_" + str(T) + ".bin"
        if force:
            clf = AdaBoostClassifier(n_estimators=T, algorithm="SAMME.R", random_state=1)
            print("LEARNING...")
            t1 = time.time()
            clf.fit(X_train, y_train)
            t2 = time.time()
            print(f"LEARNING DONE. TIME: {t2 - t1:0.3} s.")
            utils.pickle_all(clf_path + clf_name, clf)
        clf = utils.unpickle_all(clf_path + clf_name)
    else:
        clf_name = "clf_sklearn_rb_faces_n_" + str(n) + "_s_" + str(s) + "_p_" + str(p) + "_T_" + str(T) + ".bin"
        if force:
            clf = boosting.RealBoostBins(T, B)
            print("LEARNING...")
            t1 = time.time()
            clf.fit(X_train, y_train)
            t2 = time.time()
            print(f"LEARNING DONE. TIME: {t2 - t1:0.3} s.")
            utils.pickle_all(clf_path + clf_name, clf)
        clf = utils.unpickle_all(clf_path + clf_name)

    return clf

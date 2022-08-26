import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RealBoostBins(BaseEstimator, ClassifierMixin):

    def __init__(self, T, B):
        self.T_ = T
        self.B_ = B
        self.class_labels_ = None
        self.mins_ = None
        self.maxes_ = None
        self.feature_indexes_ = np.zeros(T, "int32")
        self.logits_ = np.zeros((T, B))
        self.OUTLIERS_RATIO_ = 0.05

    def logit(self, W_pos, W_neg):
        LOGIT_MAX = 2.0
        if W_pos == 0.0 and W_neg == 0:
            return 0.0
        if W_pos == 0.0:
            return -LOGIT_MAX
        if W_neg == 0.0:
            return LOGIT_MAX
        return np.clip(0.5 * np.log(W_pos / W_neg), -LOGIT_MAX, LOGIT_MAX)

    def fit(self, X, y):
        self.class_labels_ = np.unique(y)  # we assume the first class to be negative, second positive
        m, n = X.shape
        yy = np.zeros(m, "int8")
        indexes_negative = y == self.class_labels_[0]
        indexes_positive = y == self.class_labels_[1]
        yy[indexes_negative] = -1
        yy[indexes_positive] = 1

        self.mins_ = np.zeros(n)
        self.maxes_ = np.zeros(n)
        for j in range(n):
            sorted_j = np.sort(X[:, j])
            self.mins_[j] = sorted_j[
                int(np.ceil(self.OUTLIERS_RATIO_ * m))
            ]
            self.maxes_[j] = sorted_j[
                int(np.floor((1.0 - self.OUTLIERS_RATIO_) * m))
            ]
        X_binned = np.clip(
            np.int8(
                (X - self.mins_) / (self.maxes_ - self.mins_) * self.B_
            ),
            0,
            self.B_ - 1
        )

        print("PREPARING INDEXER...")
        indexer_positive = np.zeros((n, self.B_, m), dtype="bool")
        indexer_negative = np.zeros((n, self.B_, m), dtype="bool")
        for j in range(n):
            for b in range(self.B_):
                indexes_j_b = X_binned[:, j] == b
                indexer_positive[j, b] = np.logical_and(
                    indexes_j_b,
                    indexes_positive
                )
                indexer_negative[j, b] = np.logical_and(
                    indexes_j_b,
                    indexes_negative
                )
        print("PREPARING INDEXER DONE.")

        w = np.ones(m) / m
        for t in range(self.T_):
            j_best = None
            logits_best = None
            err_exp_best = np.inf
            for j in range(n):
                logits = np.zeros(self.B_)
                for b in range(self.B_):
                    W_positive = w[indexer_positive[j, b]].sum()
                    W_negative = w[indexer_negative[j, b]].sum()
                    logits[b] = self.logit(W_positive, W_negative)
                err_exp = np.sum(w * np.exp(-yy * logits[X_binned[:, j]]))
                if err_exp < err_exp_best:
                    err_exp_best = err_exp
                    logits_best = logits
                    j_best = j
            self.feature_indexes_[t] = j_best
            self.logits_[t] = logits_best
            w = w * np.exp(-yy * logits_best[X_binned[:, j_best]])
            w /= err_exp_best
            print(f"T: {t}, "
                  f"J: {j_best}, "
                  f"ERR_EXP: {err_exp_best}, "
                  f"LOGITS: {np.round(logits_best, 2)}")

    def predict(self, X):
        return self.class_labels_[(self.decision_function(X) > 0.0) * 1]

    def decision_function(self, X):
        m, n = X.shape
        X_binned = np.clip(np.int8((X - self.mins_) / (self.maxes_ - self.mins_) * self.B_), 0, self.B_ - 1)
        T = self.feature_indexes_.size
        F = np.zeros(m)
        for i in range(m):
            F[i] = self.logits_[np.arange(T), X_binned[i, self.feature_indexes_]].sum()
        return F

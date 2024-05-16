import logging
import numpy as np


class SelfTrainingClassifier(object):
    def __init__(self, base_estimator, threshold, threshold2, max_iter):
        self.base_estimator_ = base_estimator
        self.threshold = threshold
        self.threshold2 = threshold2
        self.max_iter = max_iter

    def fit(self, X, y):
        MutualInformation(kernel="gaussian", bandwidth=1.0)
        if not (0 <= self.threshold < 1):
            raise ValueError("Parameter threshold must be in the range [0, 1),"
                             f" current value is {self.threshold}")
        if not (0 <= self.threshold2 < 1):
            raise ValueError("Parameter threshold2 must be in the range [0, 1),"
                             f" current value is {self.threshold2}")

        has_label = y != -1

        if np.all(has_label):
            logging.warning("All samples in y are labeled")

        self.transduction_ = np.copy(y)
        self.labeled_iter_ = np.full_like(y, -1)
        self.labeled_iter_[has_label] = 0
        self.n_iter_ = 0

        if not hasattr(self.base_estimator_, "predict_proba"):
            msg = "base_estimator ({}) needs to implement predict_proba method to return the probability of each sample!"
            raise ValueError(msg.format(type(self.base_estimator_).__name__))

        while not np.all(has_label) and (self.max_iter is None or self.n_iter_ < self.max_iter):
            self.n_iter_ += 1
            self.base_estimator_.fit(X[has_label], self.transduction_[has_label])
            prob = self.base_estimator_.predict_proba(X[~has_label])
            pred = self.base_estimator_.classes_[np.argmax(prob, axis=1)]
            max_proba = np.max(prob, axis=1)
            selected = max_proba > self.threshold
            selected_full = np.nonzero(~has_label)[0][selected]
            self.transduction_[selected_full] = pred[selected]
            has_label[selected_full] = True
            self.labeled_iter_[selected_full] = self.n_iter_

            for idx, (sample, pseudo_label) in enumerate(zip(X[~has_label], pred)):
                # Find labeled samples with the same pseudo label
                same_label_samples = X[y == pseudo_label]
                n = 0
                same_label_sample = same_label_samples[np.random.choice(same_label_samples.shape[0], size=10, replace=False), :]
                for labeled_samples in same_label_sample:

                    temp = (omega * ncosine(np.array(sample), np.array(labeled_samples)) +
                            (1 - omega) * normalized_mutual_info_score(sample, labeled_samples))
                    n = n + 1
                avg_score = temp / n

                if avg_score >= self.threshold2:
                    self.transduction_[~has_label][idx] = pseudo_label
                else:
                    self.transduction_[~has_label][idx] = -1

            # Update the has_label array
            has_label[~has_label] = self.transduction_[~has_label] != -1

            if selected_full.shape[0] == 0:
                self.termination_condition_ = "no_change"
                break

            print(f"After iteration {self.n_iter_}, {selected_full.shape[0]} unlabeled samples have been added with new labels.")

        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"
        if np.all(has_label):
            self.termination_condition_ = "all_labeled"

        self.base_estimator_.fit(X[has_label], self.transduction_[has_label])
        self.classes_ = self.base_estimator_.classes_
        return self

    def predict(self, X):
        return self.base_estimator_.predict(X)

    def predict_proba(self, X):
        return self.base_estimator_.predict_proba(X)

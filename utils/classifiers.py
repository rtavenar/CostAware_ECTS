import numpy

from sklearn import svm

from sklearn.base import clone
from sklearn.cross_validation import StratifiedKFold

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def linear_svm(probability=False):
    try:
        return svm.SVC(kernel="linear", random_state=0, probability=probability)
    except TypeError:  # sklearn 0.11
        return svm.SVC(kernel="linear", probability=probability)


def fit_clf_at_all_times(clf_base, data, labels, min_t, params=None):
    if params is None:
        params = {}
    d_clf = {}
    n_t = data.shape[1]
    for t in range(min_t, n_t):
        new_clf = clone(clf_base)
        set_params_and_fit(new_clf, params, data[:, :t], labels)
        d_clf[t] = new_clf
    return d_clf


def set_params_and_fit(clf, params, data, labels):
    clf.set_params(**params)
    clf.fit(data, labels)
    return clf


def svm_cv(clf, Cs, data, labels, cv):
    best_acc, C_opt = 0., Cs[0]
    for C_ in Cs:
        cur_acc = 0.
        for idx_train, idx_test in StratifiedKFold(labels, cv):
            data_train, labels_train = data[idx_train], labels[idx_train]
            data_test, labels_test = data[idx_test], labels[idx_test]
            set_params_and_fit(clf, {"C": C_}, data_train, labels_train)
            cur_acc += numpy.sum(clf.predict(data_test) == labels_test) / len(labels_test)
        if cur_acc > best_acc:
            best_acc = cur_acc
            C_opt = C_
    return C_opt
import time
import numpy

from sklearn.base import clone
from sklearn.cross_validation import StratifiedKFold

from utils import classifiers, cost, rnd, expected_costs

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def decision_labels(data, labels, clf, C_curves, cv, min_t, t_cost, cost_mat, verbose=False):
    n_train_valid, n_t = data.shape[:2]
    npy_indices = numpy.arange(n_train_valid)
    curves = numpy.zeros((n_train_valid, n_t))
    n_classes = cost_mat.shape[0]
    for idx_train, idx_valid in StratifiedKFold(labels, cv):
        data_train, labels_train = data[idx_train], labels[idx_train]
        data_valid, labels_valid = data[idx_valid], labels[idx_valid]
        pred_errors = {}
        for t in range(min_t, n_t):
            new_clf = clone(clf)
            classifiers.set_params_and_fit(new_clf, {"C": C_curves}, data_train[:, :t], labels_train)
            predicted = new_clf.predict_proba(data_valid[:, :t])
            for idx, probas, y in zip(npy_indices[idx_valid], predicted, labels_valid):
                for y_hat in range(n_classes):
                    pred_errors[t, idx, y, y_hat] = probas[y_hat]
        curves += expected_costs.cost_curves(pred_errors, t_cost, cost_mat, min_t, n_train_valid)
    if verbose:
        numpy.savetxt("../results/curves_2step.txt", curves)
    dec_labels = prediction_decision(curves, data, min_t, n_t)
    return dec_labels


def fit_2step_classifiers(data, labels, dec_labels, clf, C_decision, C_classif, min_t):
    h_decision = {}
    h_classif = {}
    n_t = data.shape[1]
    for t in range(min_t, n_t):
        h_decision[t] = clone(clf)
        h_classif[t] = clone(clf)
        try:
            classifiers.set_params_and_fit(h_decision[t], {"C": C_decision}, data[:, :t], dec_labels[t])
            classifiers.set_params_and_fit(h_classif[t], {"C": C_classif}, data[:, :t], labels)
        except ValueError:
            h_decision[t] = dec_labels[t][0]
            classifiers.set_params_and_fit(h_classif[t], {"C": C_classif}, data[:, :t], labels)
    return h_decision, h_classif


def prepare_2step_classifiers(data, labels, clf, C_curves, C_decision, C_classif, cv, min_t, t_cost, cost_mat,
                              verbose=False):
    dec_labels = decision_labels(data, labels, clf, C_curves, cv, min_t, t_cost, cost_mat, verbose=verbose)
    return fit_2step_classifiers(data, labels, dec_labels, clf, C_decision, C_classif, min_t)


def predict_2step(h_decision, h_classif, min_t, data):
    n_test, n_t = data.shape[:2]
    predicted = numpy.zeros((n_test,), dtype=numpy.int32) - 1
    tau = numpy.zeros((n_test,), dtype=numpy.int32) - 1
    for t in range(min_t, n_t):
        if type(h_decision[t]) == numpy.int32:
            if h_decision[t] == 0:
                continue
            else:
                for idx in range(n_test):
                    if predicted[idx] < 0:
                        predicted[idx] = h_classif[t].predict(data[idx, :t])
                        tau[idx] = t
        else:
            dec = h_decision[t].predict(data[:, :t])
            for idx in range(n_test):
                if dec[idx] == 1 and predicted[idx] < 0:
                    predicted[idx] = h_classif[t].predict(data[idx, :t])
                    tau[idx] = t
        if numpy.alltrue(predicted >= 0):
            break
    return predicted, tau


def cost_2step(data, labels, min_t, t_cost, cost_mat, h_decision, h_classif, verbose=False):
    start_time = time.clock()
    predicted, tau = predict_2step(h_decision, h_classif, min_t, data)
    total_time = time.clock() - start_time
    tot_cost, acc, tau_bar = cost.total_cost(predicted, labels, tau, cost_mat, t_cost)
    if verbose:
        numpy.savetxt("../results/tau_2step.txt", tau)
    return tot_cost, acc, tau_bar, total_time


def cvopt_2step(clf, Cs, data_train_valid, labels_train_valid, data_test, labels_test, cv, min_t, t_cost, cost_mat):
    best_cost, C_opt = numpy.inf, (None, None, None)
    for C_curves in Cs:
        for C_decision in Cs:
            for C_classif in Cs:
                h_decision, h_classif = prepare_2step_classifiers(data_train_valid, labels_train_valid, clf, C_curves,
                                                                  C_decision, C_classif, cv, min_t, t_cost, cost_mat)
                tot_cost, _, _, _ = cost_2step(data_test, labels_test, min_t, t_cost, cost_mat, h_decision, h_classif)
                if tot_cost < best_cost:
                    best_cost = tot_cost
                    C_opt = C_curves, C_decision, C_classif
    return C_opt


def prediction_decision(curves, data, min_t, n_t):
    gamma = numpy.zeros((n_t, data.shape[0]), dtype=numpy.int32) - 1
    for t in range(min_t, n_t):
        gamma[t] = (curves[:, t] <= numpy.min(curves[:, t:], axis=1))
    return gamma


def cost_cv(data_train_valid, labels_train_valid, data_test, labels_test, clf, Cs, cv, min_t, train_indices, t_cost,
            cost_mat, verbose=False):
    use_proba = clf.get_params().get("probability", False)
    assert use_proba, "Probability estimates should be activated"

    valid_indices = rnd.opposite_indices(train_indices, data_train_valid.shape[0])
    data_train, labels_train = data_train_valid[train_indices], labels_train_valid[train_indices]
    data_valid, labels_valid = data_train_valid[valid_indices], labels_train_valid[valid_indices]

    start_time = time.clock()
    C_curves, C_decision, C_classif = cvopt_2step(clf, Cs, data_train, labels_train, data_valid, labels_valid, cv,
                                                  min_t, t_cost, cost_mat)
    h_decision, h_classif = prepare_2step_classifiers(data_train_valid, labels_train_valid, clf, C_curves, C_decision,
                                                      C_classif, cv, min_t, t_cost, cost_mat, verbose=verbose)
    train_time = time.clock() - start_time
    tot_cost, acc, tau_bar, test_time = cost_2step(data_test, labels_test, min_t, t_cost, cost_mat, h_decision,
                                                   h_classif, verbose=verbose)
    return tot_cost, acc, tau_bar, train_time, test_time
import time
import numpy

from sklearn.cross_validation import StratifiedKFold

from utils import cluster, cost, classifiers, expected_costs, rnd

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def expected_errors(data, labels, clustering, clf, C, cv, min_t):
    pred_errors = {}
    use_proba = clf.get_params().get("probability", False)
    sums = {}
    n_t = data.shape[1]
    n_classes = len(set(labels))
    for t in range(min_t, n_t):
        for idx_train, idx_valid in StratifiedKFold(labels, cv):
            classifiers.set_params_and_fit(clf, {"C": C}, data[idx_train, :t], labels[idx_train])
            if use_proba:
                predicted = clf.predict_proba(data[idx_valid, :t])
                for c_k, y, probas in zip(clustering.labels_[idx_valid], labels[idx_valid], predicted):
                    for y_hat in range(n_classes):
                        pred_errors[t, c_k, y, y_hat] = pred_errors.get((t, c_k, y, y_hat), 0) + probas[y_hat]
                    sums[t, c_k, y] = sums.get((t, c_k, y), 0) + 1
            else:
                predicted = clf.predict(data[idx_valid, :t])
                for c_k, y, y_hat in zip(clustering.labels_[idx_valid], labels[idx_valid], predicted):
                    if y != y_hat:
                        pred_errors[t, c_k, y, y_hat] = pred_errors.get((t, c_k, y, y_hat), 0) + 1
                    sums[t, c_k, y] = sums.get((t, c_k, y), 0) + 1
    for t, c_k, y, y_hat in pred_errors.keys():
        pred_errors[t, c_k, y, y_hat] /= sums[t, c_k, y]
    return pred_errors


def prediction(data, clustering, curves, d_clf, min_t=4, lbda=1., verbose=False):
    n_test, n_t = data.shape[:2]
    predicted_outputs = numpy.zeros((n_test,), dtype=numpy.int32) - 1
    predicted_timings = numpy.zeros((n_test,), dtype=numpy.int32) - 1
    if verbose:
        curves_test = numpy.zeros((n_test, n_t)) * numpy.nan
        t_target = 40
        delta_t = numpy.zeros((n_test, n_t)) * numpy.nan
    else:
        t_target, curves_test = None, None
    for i_test in range(n_test):
        for t in range(min_t, n_t):
            pred_costs = expected_costs.adaptive_cost_curve(data[i_test, :t], clustering.cluster_centers_[:, :t],
                                                            curves[:, t:], lbda)
            if verbose:
                if t == t_target:
                    curves_test[i_test, t:] = pred_costs
                delta_t[i_test, t] = t + numpy.argmin(pred_costs)
            if numpy.argmin(pred_costs) == 0:
                predicted_outputs[i_test] = d_clf[t].predict(data[i_test, :t])
                predicted_timings[i_test] = t
                break
        if verbose:
            delta_t[i_test] -= predicted_timings[i_test]
    if verbose:
        if clustering.__class__.__name__ == "KMeans":
            numpy.savetxt("../results/delta_t_baseline.txt", delta_t / n_t)
            numpy.savetxt("../results/curves_baseline_test.txt", curves_test)
        else:
            numpy.savetxt("../results/delta_t_nocluster.txt", delta_t / n_t)
            numpy.savetxt("../results/curves_nocluster_test.txt", curves_test)
    return predicted_outputs, predicted_timings


def cost_1run(d_clf, data, labels, clustering, curves, cost_mat, t_cost, min_t=4, lbda=1., verbose=False):
    start_time = time.clock()
    pred_class, pred_tau = prediction(data, clustering, curves, d_clf, min_t=min_t, lbda=lbda, verbose=verbose)
    total_time = time.clock() - start_time
    tot_cost, acc, tau_bar = cost.total_cost(pred_class, labels, pred_tau, cost_mat, t_cost)
    if verbose:
        if clustering.__class__.__name__ == "KMeans":
            numpy.savetxt("../results/tau_baseline.txt", pred_tau)
            numpy.savetxt("../results/curves_baseline.txt", curves)
        else:
            numpy.savetxt("../results/tau_nocluster.txt", pred_tau)
            numpy.savetxt("../results/curves_nocluster.txt", curves)
    return tot_cost, acc, tau_bar, total_time


def cvopt_baseline(clf, Cs, data_train_valid, labels_train_valid, data_test, labels_test, lambdas, clustering, cost_mat,
                   t_cost, min_t, cv):
    min_cost, param_opt = numpy.inf, (None, None, None)
    for C_curves in Cs:
        for C_classif in Cs:
            params_svm = {"C": C_classif}
            for lambda_ in lambdas:
                pred_errors = expected_errors(data_train_valid, labels_train_valid, clustering, clf, C_curves, cv,
                                              min_t)
                curves = expected_costs.cost_curves(pred_errors, t_cost, cost_mat, min_t,
                                                    cluster.n_clusters(clustering))
                clf.set_params(**params_svm)
                d_clf = classifiers.fit_clf_at_all_times(clf, data_train_valid, labels_train_valid, min_t)
                tot_cost = cost_1run(d_clf, data=data_test, labels=labels_test, clustering=clustering, curves=curves,
                                     cost_mat=cost_mat, t_cost=t_cost, min_t=min_t, lbda=lambda_)[0]
                if tot_cost < min_cost:
                    param_opt = lambda_, C_curves, C_classif
                    min_cost = tot_cost
    return param_opt


def cost_cv(data_train_valid, labels_train_valid, data_test, labels_test, train_indices, clf, Cs, lambdas, cv,
            min_t, t_cost, cost_mat, clustering=None, verbose=False):
    n_train, n_t = data_train_valid.shape[:2]

    valid_indices = rnd.opposite_indices(train_indices, data_train_valid.shape[0])
    data_train, labels_train = data_train_valid[train_indices], labels_train_valid[train_indices]
    data_valid, labels_valid = data_train_valid[valid_indices], labels_train_valid[valid_indices]

    # 1- Training
    start_time = time.clock()
    if clustering is None:
        clustering = cluster.adaptive_kmeans(data_train_valid, k_values=range(5, min(n_train, 100), 5))

    lambda_opt, C_curves, C_classif = cvopt_baseline(clf, Cs, data_train, labels_train, data_valid, labels_valid,
                                                     lambdas, clustering, cost_mat, t_cost, min_t, cv)
    pred_errors = expected_errors(data_train_valid, labels_train_valid, clustering, clf, C_curves, cv, min_t)
    curves = expected_costs.cost_curves(pred_errors, t_cost, cost_mat, min_t, cluster.n_clusters(clustering))
    d_clf = classifiers.fit_clf_at_all_times(clf, data_train_valid, labels_train_valid, min_t, params={"C": C_classif})
    train_time = time.clock() - start_time

    # 2- Test
    tot_cost, acc, tau_bar, test_time = cost_1run(d_clf, data_test, labels_test, clustering, curves=curves,
                                                  cost_mat=cost_mat, t_cost=t_cost, min_t=min_t, lbda=lambda_opt,
                                                  verbose=verbose)
    return tot_cost, acc, tau_bar, train_time, test_time
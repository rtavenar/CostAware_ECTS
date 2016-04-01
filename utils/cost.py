import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def total_cost(predicted, labels, tau, cost_mat, t_cost):
    n_test = labels.shape[0]
    tau_bar = numpy.mean(tau)
    cost = numpy.sum(t_cost[tau])
    for lab, pred in zip(labels, predicted):
        cost += cost_mat[lab, pred]
    acc = float(numpy.sum(predicted == labels)) / n_test
    return cost, acc, tau_bar


import numpy
from scipy.spatial.distance import cdist

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def cost_curves(pred_errors, t_cost, cost_mat, min_t, n):
    # NB: here n can either be the number of classes, hence pred_error keys are made of
    # (timeindex, classidx, groundtruthY)
    # or the number of training time series and pred_error keys are then made of
    # (timeindex, timeseriesidx, groundtruthY)
    curves = numpy.tile(t_cost, (n, 1))
    curves[:, :min_t] = numpy.inf
    for (t, idx, y, y_hat), probas in pred_errors.items():
        curves[idx, t] += probas * cost_mat[y, y_hat]
    return curves


def adaptive_cost_curve(target_ts, training_ts, training_curves, lbda=1.):
    distances = cdist(target_ts.reshape((1, -1)), training_ts).reshape((-1, ))
    delta = (numpy.mean(distances) - distances) / numpy.mean(distances)
    prob_distances = 1. / (1. + numpy.exp(- lbda * delta))
    prob_distances /= numpy.sum(prob_distances)
    return numpy.sum(numpy.multiply(training_curves, prob_distances[:, numpy.newaxis]), axis=0)
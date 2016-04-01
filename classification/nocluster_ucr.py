import sys

import numpy

from classification.methods import method_baseline
from datasets import UCRreader
from utils import rnd, classifiers, PseudoClustering

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

min_t = 4
cv = 3
Cs = numpy.logspace(-1, 1, 5)
lambdas = numpy.logspace(1, 3, 5)
clf = classifiers.linear_svm(probability=True)
verbose = False

if len(sys.argv) > 1:
    cl_dataset = sys.argv[1]
    cl_tcf = float(sys.argv[2])
else:
    cl_dataset = "Gun_Point"
    cl_tcf = .0005

print("method;dataset;C(t)/t;cost;accuracy;tau_bar;train_time;test_time")
data_train, labels_train, data_test, labels_test = UCRreader.train_test(cl_dataset)
assert UCRreader.get_min_pop_per_class(labels_train) >= 10, "Skipping dataset %s: not enough data\n" % cl_dataset
n_train, n_t = data_train.shape
max_class_id = numpy.max(labels_train)
cost_mat = numpy.ones((max_class_id + 1, max_class_id + 1)) - numpy.eye(max_class_id + 1)
train_indices = rnd.random_indices(n_train, int(n_train / 2))
t_cost = cl_tcf * numpy.arange(n_t)

clustering = PseudoClustering.PseudoClustering(dataset=data_train)
cost, acc, tau_bar, train_time, test_time = method_baseline.cost_cv(data_train, labels_train, data_test, labels_test,
                                                                    train_indices, clf, Cs, lambdas, cv, min_t, t_cost,
                                                                    cost_mat, clustering=clustering, verbose=verbose)
print("nocluster;%s;%.4f;%.3f;%.3f;%.3f;%.3f;%.3f" % (cl_dataset, cl_tcf, cost, acc, tau_bar, train_time, test_time))

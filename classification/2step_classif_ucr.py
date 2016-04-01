import sys
import numpy

from classification.methods import method_2step
from datasets import UCRreader
from utils import rnd, classifiers

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

min_t = 4
cv = 3
Cs = numpy.logspace(-1, 1, 5)
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
cost, acc, tau_bar, train_time, test_time = method_2step.cost_cv(data_train, labels_train, data_test, labels_test,
                                                                 clf, Cs, cv, min_t, train_indices, t_cost, cost_mat,
                                                                 verbose=verbose)
print("2step;%s;%.4f;%.3f;%.3f;%.3f;%.3f;%.3f" % (cl_dataset, cl_tcf, cost, acc, tau_bar, train_time, test_time))

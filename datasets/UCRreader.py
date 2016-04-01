import os
import numpy
from scipy.io import loadmat

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def per_individual_normalize_01(data):
    for f in data:
        f -= numpy.nanmin(f)
        f /= numpy.nanmax(f)
    return data


def correct_labels(labels):
    if numpy.sum(labels == 0) == 0:
        labels[labels == -1] = 0
    labels -= labels.min()
    return labels


def read(filename, normalize=False):
    try:
        raw_data = numpy.loadtxt(filename)
    except ValueError:
        raw_data = numpy.loadtxt(filename, delimiter=",")
    labels = raw_data[:, 0].astype(numpy.int32)
    labels = correct_labels(labels)
    if normalize:
        raw_data = per_individual_normalize_01(raw_data[:, 1:])
    else:
        raw_data = raw_data[:, 1:]
    return raw_data, labels


def read_mat(filename, normalize=False):
    d_raw_data = loadmat(filename)
    raw_data = None
    #print(filename, d_raw_data.keys())
    for key in d_raw_data.keys():
        if "train" in key.lower() or "test" in key.lower():
            raw_data = d_raw_data[key]
    if raw_data is None:
        raise KeyError
    labels = raw_data[:, 0].astype(numpy.int32)
    labels = correct_labels(labels)
    if normalize:
        raw_data = per_individual_normalize_01(raw_data[:, 1:])
    else:
        raw_data = raw_data[:, 1:]
    return raw_data, labels


def train_test(dataset_name, path="../datasets/ucr/"):
    basedir = os.path.join(path, dataset_name)
    fname_train = os.path.join(basedir, "%s_TRAIN" % dataset_name)
    if not os.path.exists(fname_train) and os.path.exists(fname_train + ".mat"):
        fname_train += ".mat"
        data_train, labels_train = read_mat(fname_train)
    else:
        data_train, labels_train = read(fname_train)
    fname_test = os.path.join(basedir, "%s_TEST" % dataset_name)
    if not os.path.exists(fname_test) and os.path.exists(fname_test + ".mat"):
        fname_test += ".mat"
        data_test, labels_test = read_mat(fname_test)
    else:
        data_test, labels_test = read(fname_test)
    return data_train, labels_train, data_test, labels_test


def list_datasets(path="../datasets/ucr/"):
    datasets = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))
                       and f != ".svn"])
    for d in ["OSULeaf", "FaceAll"]:
        datasets.remove(d)
        datasets.append(d)
    return datasets


def get_min_pop_per_class(labels):
    lst_pop = [int(numpy.sum(labels == k)) for k in set(labels)]
    return min(lst_pop)


if __name__ == "__main__":
    for ds_name in list_datasets("ucr/"):
        data_train, labels_train, data_test, labels_test = train_test(ds_name)
        print(ds_name, set(labels_train), set(labels_test))
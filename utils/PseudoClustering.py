from sklearn.cluster import KMeans
import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class PseudoClustering(KMeans):
    def __init__(self, dataset, labels=None, inertia=0.):
        KMeans.__init__(self)
        self.cluster_centers_ = dataset
        self.n_clusters = dataset.shape[0]
        if labels is None:
            self.labels_ = numpy.arange(dataset.shape[0]).astype(numpy.int32)
        else:
            self.labels_ = labels
        self.inertia_ = inertia


def up_to_t(clustering, t):
    return PseudoClustering(clustering.cluster_centers_[:, :t], labels=clustering.labels_, inertia=clustering.inertia_)
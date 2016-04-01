import warnings

from sklearn.cluster import KMeans
from sklearn import metrics

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def adaptive_kmeans(data, k_values):
    k = pick_k_kmeans(data=data, k_values=k_values)
    try:
        clustering = KMeans(n_clusters=k, init="k-means++", n_init=10)
    except TypeError:
        clustering = KMeans(k=k, init="k-means++", n_init=10)
    clustering.fit(data)
    return clustering


def pick_k_kmeans(data, k_values):
    k_max = k_values[0]
    sil_max = -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for k in k_values:
            try:
                kmeans_model = KMeans(n_clusters=k).fit(data)
            except TypeError:
                kmeans_model = KMeans(k=k).fit(data)
            sil = metrics.silhouette_score(data, kmeans_model.labels_, metric='euclidean')
            if sil > sil_max:
                k_max = k
                sil_max = sil
    return k_max


def n_clusters(clustering):
    return clustering.cluster_centers_.shape[0]


if __name__ == "__main__":
    from datasets import SyntheticDataset

    ds = SyntheticDataset.SyntheticDataset.load_from_folder("../datasets/synthetic/dataset_b0.02_noise0.1")
    print(pick_k_kmeans(ds.data_train, range(5, 100, 5)))

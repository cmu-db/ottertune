#
# OtterTune - cluster.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Jul 4, 2016

@author: dva
'''

from abc import ABCMeta, abstractproperty
from collections import OrderedDict

import copy
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans as SklearnKMeans

from .base import ModelBase


class KMeans(ModelBase):
    """
    KMeans:

    Fits an Sklearn KMeans model to X.


    See also
    --------
    http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


    Attributes
    ----------
    n_clusters_ : int
                  The number of clusters, K

    cluster_inertia_ : float
                       Sum of squared distances of samples to their closest cluster center

    cluster_labels_ : array, [n_clusters_]
                      Labels indicating the membership of each point

    cluster_centers_ : array, [n_clusters, n_features]
                       Coordinates of cluster centers

    sample_labels_ : array, [n_samples]
                     Labels for each of the samples in X

    sample_distances_ : array, [n_samples]
                        The distance between each sample point and its cluster's center


    Constants
    ---------
    SAMPLE_CUTOFF_ : int
                     If n_samples > SAMPLE_CUTOFF_ then sample distances
                     are NOT recorded
    """

    SAMPLE_CUTOFF_ = 1000

    def __init__(self):
        self.model_ = None
        self.n_clusters_ = None
        self.sample_labels_ = None
        self.sample_distances_ = None

    @property
    def cluster_inertia_(self):
        # Sum of squared distances of samples to their closest cluster center
        return None if self.model_ is None else \
            self.model_.inertia_

    @property
    def cluster_labels_(self):
        # Cluster membership labels for each point
        return None if self.model_ is None else \
            copy.deepcopy(self.model_.labels_)

    @property
    def cluster_centers_(self):
        # Coordinates of the cluster centers
        return None if self.model_ is None else \
            copy.deepcopy(self.model_.cluster_centers_)

    def _reset(self):
        """Resets all attributes (erases the model)"""
        self.model_ = None
        self.n_clusters_ = None
        self.sample_labels_ = None
        self.sample_distances_ = None

    def fit(self, X, K, sample_labels=None, estimator_params=None):
        """Fits a Sklearn KMeans model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        K : int
            The number of clusters.

        sample_labels : array-like, shape (n_samples), optional
                        Labels for each of the samples in X.

        estimator_params : dict, optional
                           The parameters to pass to the KMeans estimators.


        Returns
        -------
        self
        """
        self._reset()
        # Note: previously set n_init=50
        self.model_ = SklearnKMeans(K)
        if estimator_params is not None:
            assert isinstance(estimator_params, dict)
            self.model_.set_params(**estimator_params)

        # Compute Kmeans model
        self.model_.fit(X)
        if sample_labels is None:
            sample_labels = ["sample_{}".format(i) for i in range(X.shape[0])]
        assert len(sample_labels) == X.shape[0]
        self.sample_labels_ = np.array(sample_labels)
        self.n_clusters_ = K

        # Record sample label/distance from its cluster center
        self.sample_distances_ = OrderedDict()
        for cluster_label in range(self.n_clusters_):
            assert cluster_label not in self.sample_distances_
            member_rows = X[self.cluster_labels_ == cluster_label, :]
            member_labels = self.sample_labels_[self.cluster_labels_ == cluster_label]
            centroid = np.expand_dims(self.cluster_centers_[cluster_label], axis=0)

            # "All clusters must have at least 1 member!"
            if member_rows.shape[0] == 0:
                return None

            # Calculate distance between each member row and the current cluster
            dists = np.empty(member_rows.shape[0])
            dist_labels = []
            for j, (row, label) in enumerate(zip(member_rows, member_labels)):
                dists[j] = cdist(np.expand_dims(row, axis=0), centroid, "euclidean").squeeze()
                dist_labels.append(label)

            # Sort the distances/labels in ascending order
            sort_order = np.argsort(dists)
            dists = dists[sort_order]
            dist_labels = np.array(dist_labels)[sort_order]
            self.sample_distances_[cluster_label] = {
                "sample_labels": dist_labels,
                "distances": dists,
            }
        return self

    def get_closest_samples(self):
        """Returns a list of the labels of the samples that are located closest
           to their cluster's center.


        Returns
        ----------
        closest_samples : list
                  A list of the sample labels that are located the closest to
                  their cluster's center.
        """
        if self.sample_distances_ is None:
            raise Exception("No model has been fit yet!")

        return [samples['sample_labels'][0] for samples in self.sample_distances_.values()]


class KMeansClusters(ModelBase):

    """
    KMeansClusters:

    Fits a KMeans model to X for clusters in the range [min_cluster_, max_cluster_].


    Attributes
    ----------
    min_cluster_ : int
                   The minimum cluster size to fit a KMeans model to

    max_cluster_ : int
                   The maximum cluster size to fit a KMeans model to

    cluster_map_ : dict
                   A dictionary mapping the cluster size (K) to the KMeans
                   model fitted to X with K clusters

    sample_labels_ : array, [n_samples]
                     Labels for each of the samples in X
    """

    def __init__(self):
        self.min_cluster_ = None
        self.max_cluster_ = None
        self.cluster_map_ = None
        self.sample_labels_ = None

    def _reset(self):
        """Resets all attributes (erases the model)"""
        self.min_cluster_ = None
        self.max_cluster_ = None
        self.cluster_map_ = None
        self.sample_labels_ = None

    def fit(self, X, min_cluster, max_cluster, sample_labels=None, estimator_params=None):
        """Fits a KMeans model to X for each cluster in the range [min_cluster, max_cluster].

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        min_cluster : int
                      The minimum cluster size to fit a KMeans model to.

        max_cluster : int
                      The maximum cluster size to fit a KMeans model to.

        sample_labels : array-like, shape (n_samples), optional
                        Labels for each of the samples in X.

        estimator_params : dict, optional
                           The parameters to pass to the KMeans estimators.


        Returns
        -------
        self
        """
        self._reset()
        self.min_cluster_ = min_cluster
        self.max_cluster_ = max_cluster
        self.cluster_map_ = {}
        if sample_labels is None:
            sample_labels = ["sample_{}".format(i) for i in range(X.shape[1])]
        self.sample_labels_ = sample_labels
        for K in range(self.min_cluster_, self.max_cluster_ + 1):
            tmp = KMeans().fit(X, K, self.sample_labels_, estimator_params)
            if tmp is None:  # Set maximum cluster
                assert K > min_cluster, "min_cluster is too large for the model"
                self.max_cluster_ = K - 1
                break
            else:
                self.cluster_map_[K] = tmp

        return self


class KSelection(ModelBase):
    """KSelection:

    Abstract class for techniques that approximate the optimal
    number of clusters (K).


    Attributes
    ----------
    optimal_num_clusters_ : int
                            An estimation of the optimal number of clusters K for
                            a KMeans model fit to X
    clusters_ : array, [n_clusters]
                The sizes of the clusters

    name_ : string
            The name of this technique
    """

    NAME_ = None

    __metaclass__ = ABCMeta

    def __init__(self):
        self.optimal_num_clusters_ = None
        self.clusters_ = None

    def _reset(self):
        """Resets all attributes (erases the model)"""
        self.optimal_num_clusters_ = None
        self.clusters_ = None

    @abstractproperty
    def name_(self):
        pass


class GapStatistic(KSelection):
    """GapStatistic:

    Approximates the optimal number of clusters (K).


    References
    ----------
    https://web.stanford.edu/~hastie/Papers/gap.pdf


    Attributes
    ----------
    optimal_num_clusters_ : int
                            An estimation of the optimal number of clusters K for
                            a KMeans model fit to X

    clusters_ : array, [n_clusters]
                The sizes of the clusters

    name_ : string
            The name of this technique

    log_wks_ : array, [n_clusters]
               The within-dispersion measures of X (log)

    log_wkbs_ : array, [n_clusters]
                The within-dispersion measures of the generated
                reference data sets

    khats_ : array, [n_clusters]
             The gap-statistic for each cluster
    """

    NAME_ = "gap-statistic"

    def __init__(self):
        super(GapStatistic, self).__init__()
        self.log_wks_ = None
        self.log_wkbs_ = None
        self.khats_ = None

    @property
    def name_(self):
        return self.NAME_

    def _reset(self):
        """Resets all attributes (erases the model)"""
        super(GapStatistic, self)._reset()
        self.log_wks_ = None
        self.log_wkbs_ = None
        self.khats_ = None

    def fit(self, X, cluster_map, n_b=50):
        """Estimates the optimal number of clusters (K) for a
           KMeans model trained on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        cluster_map_ : dict
                       A dictionary mapping each cluster size (K) to the KMeans
                       model fitted to X with K clusters

        n_b : int
              The number of reference data sets to generate


        Returns
        -------
        self
        """
        self._reset()
        mins, maxs = GapStatistic.bounding_box(X)
        n_clusters = len(cluster_map)

        # Dispersion for real distribution
        log_wks = np.zeros(n_clusters)
        log_wkbs = np.zeros(n_clusters)
        sk = np.zeros(n_clusters)
        for indk, (K, model) in enumerate(sorted(cluster_map.iteritems())):
            log_wks[indk] = np.log(GapStatistic.Wk(X,
                                                   model.cluster_centers_,
                                                   model.cluster_labels_))

            # Create B reference datasets
            log_b_wkbs = np.zeros(n_b)
            for i in range(n_b):
                Xb = np.empty_like(X)
                for j in range(X.shape[1]):
                    Xb[:, j] = np.random.uniform(mins[j], maxs[j], size=X.shape[0])
                Xb_model = KMeans().fit(Xb, K)
                log_b_wkbs[i] = np.log(GapStatistic.Wk(Xb,
                                                       Xb_model.cluster_centers_,
                                                       Xb_model.cluster_labels_))
            log_wkbs[indk] = sum(log_b_wkbs) / n_b
            sk[indk] = np.sqrt(sum((log_b_wkbs - log_wkbs[indk])**2) / n_b)
        sk = sk * np.sqrt(1 + 1.0 / n_b)

        khats = np.zeros(n_clusters)
        gaps = log_wkbs - log_wks
        gsks = gaps - sk
        self.clusters_ = np.array(sorted(cluster_map.keys()))
        for i in range(1, n_clusters):
            khats[i] = gaps[i - 1] - gsks[i]
            if self.optimal_num_clusters_ is None and gaps[i - 1] >= gsks[i]:
                self.optimal_num_clusters_ = self.clusters_[i - 1]
        if self.optimal_num_clusters_ is None:
            self.optimal_num_clusters_ = 0

        self.log_wks_ = log_wks
        self.log_wkbs_ = log_wkbs
        self.khats_ = khats
        return self

    @staticmethod
    def bounding_box(X):
        """Computes the box that tightly bounds X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.


        Returns
        -------
        The mins and maxs that make up the bounding box
        """
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        return mins, maxs

    @staticmethod
    def Wk(X, mu, cluster_labels):
        """Computes the within-dispersion of each cluster size (k)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        mu : array-like, shape (n_clusters, n_features)
            Coordinates of cluster centers

        cluster_labels: array-like, shape (n_samples)
                        Labels for each of the samples in X.


        Returns
        -------
        The within-dispersion of each cluster (K)
        """
        K = len(mu)
        return sum([np.linalg.norm(mu[i] - x)**2
                    for i in range(K)
                    for x in X[cluster_labels == i]])


class DetK(KSelection):
    """DetK:

    Approximates the optimal number of clusters (K).


    References
    ----------
    https://www.ee.columbia.edu/~dpwe/papers/PhamDN05-kmeans.pdf


    Attributes
    ----------
    optimal_num_clusters_ : int
                            An estimation of the optimal number of clusters K for
                            KMeans models fit to X

    clusters_ : array, [n_clusters]
                The sizes of the clusters

    name_ : string
            The name of this technique

    fs_ : array, [n_clusters]
          The computed evaluation functions F(K) for each cluster size K
    """

    NAME_ = "det-k"

    def __init__(self):
        super(DetK, self).__init__()
        self.fs_ = None

    @property
    def name_(self):
        return DetK.NAME_

    def _reset(self):
        """Resets all attributes (erases the model)"""
        super(DetK, self)._reset()
        self.fs_ = None

    def fit(self, X, cluster_map):
        """Estimates the optimal number of clusters (K) for a
           KMeans model trained on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        cluster_map_ : dict
                       A dictionary mapping each cluster size (K) to the KMeans
                       model fitted to X with K clusters


        Returns
        -------
        self
        """
        self._reset()
        n_clusters = len(cluster_map)
        nd = X.shape[1]
        fs = np.empty(n_clusters)
        sks = np.empty(n_clusters)
        alpha = {}
        # K from 1 to maximum_cluster_
        for i, (K, model) \
                in enumerate(sorted(cluster_map.iteritems())):
            # Compute alpha(K, nd) (i.e. alpha[K])
            if K == 2:
                alpha[K] = 1 - 3.0 / (4 * nd)
            elif K > 2:
                alpha[K] = alpha[K - 1] + (1 - alpha[K - 1]) / 6.0
            sks[i] = model.cluster_inertia_

            if K == 1:
                fs[i] = 1
            elif sks[i - 1] == 0:
                fs[i] = 1
            else:
                fs[i] = sks[i] / (alpha[K] * sks[i - 1])
        self.clusters_ = np.array(sorted(cluster_map.keys()))
        self.optimal_num_clusters_ = self.clusters_[np.argmin(fs)]
        self.fs_ = fs
        return self


def create_kselection_model(model_name):
    """Constructs the KSelection model object with the given name

    Parameters
    ----------
    model_name : string
                 Name of the KSelection model.
                 One of ['gap-statistic', 'det-k']


    Returns
    -------
    The constructed model object
    """
    kselection_map = {
        DetK.NAME_: DetK,
        GapStatistic.NAME_: GapStatistic,
    }
    if model_name not in kselection_map:
        raise Exception("KSelection model {} not supported!".format(model_name))
    else:
        return kselection_map[model_name]()

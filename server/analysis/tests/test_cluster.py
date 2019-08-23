#
# OtterTune - test_cluster.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import unittest
import numpy as np
from sklearn import datasets

from analysis.cluster import KMeans, KMeansClusters, create_kselection_model


class TestKMeans(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestKMeans, cls).setUpClass()
        iris = datasets.load_iris()
        cls.model = KMeans()
        cls.model.fit(iris.data, 5, iris.target,
                      estimator_params={'n_init': 50, 'random_state': 42})

    def test_kmeans_n_clusters(self):
        self.assertEqual(self.model.n_clusters_, 5)

    def test_kmeans_cluster_inertia(self):
        self.assertAlmostEqual(self.model.cluster_inertia_, 46.535, 2)

    def test_kmeans_cluster_labels(self):
        expected_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 2, 3, 3, 3,
                           2, 3, 2, 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2, 3, 3, 3, 3,
                           3, 3, 3, 2, 2, 2, 2, 3, 2, 3, 3, 3, 2, 2, 2, 3, 2, 2, 2,
                           2, 2, 3, 2, 2, 4, 3, 0, 4, 4, 0, 2, 0, 4, 0, 4, 4, 4, 3,
                           4, 4, 4, 0, 0, 3, 4, 3, 0, 3, 4, 0, 3, 3, 4, 0, 0, 0, 4,
                           3, 3, 0, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 3]
        for lab_actual, lab_expected in zip(self.model.cluster_labels_, expected_labels):
            self.assertEqual(lab_actual, lab_expected)

    def test_kmeans_sample_labels(self):
        for lab_actual, lab_expected in zip(self.model.sample_labels_, datasets.load_iris().target):
            self.assertEqual(lab_actual, lab_expected)

    def test_kmeans_cluster_centers(self):
        expected_centers = [[7.475, 3.125, 6.300, 2.050],
                            [5.006, 3.418, 1.464, 0.244],
                            [5.508, 2.600, 3.908, 1.204],
                            [6.207, 2.853, 4.746, 1.564],
                            [6.529, 3.058, 5.508, 2.162]]
        for row_actual, row_expected in zip(self.model.cluster_centers_, expected_centers):
            for val_actual, val_expected in zip(row_actual, row_expected):
                self.assertAlmostEqual(val_actual, val_expected, 2)


class TestKSelection(unittest.TestCase):

    def setUp(self):
        np.random.seed(seed=42)

    @classmethod
    def setUpClass(cls):
        super(TestKSelection, cls).setUpClass()

        # Load Iris data
        iris = datasets.load_iris()
        cls.matrix = iris.data
        cls.kmeans_models = KMeansClusters()
        cls.kmeans_models.fit(cls.matrix,
                              min_cluster=1,
                              max_cluster=10,
                              sample_labels=iris.target,
                              estimator_params={'n_init': 50, 'random_state': 42})

    def test_detk_optimal_num_clusters(self):
        # Compute optimal # cluster using det-k
        detk = create_kselection_model("det-k")
        detk.fit(self.matrix, self.kmeans_models.cluster_map_)
        self.assertEqual(detk.optimal_num_clusters_, 2)

    def test_gap_statistic_optimal_num_clusters(self):
        # Compute optimal # cluster using gap-statistics
        gap = create_kselection_model("gap-statistic")
        gap.fit(self.matrix, self.kmeans_models.cluster_map_)
        self.assertEqual(gap.optimal_num_clusters_, 8)

    def test_silhouette_optimal_num_clusters(self):
        # Compute optimal # cluster using Silhouette Analysis
        sil = create_kselection_model("s-score")
        sil.fit(self.matrix, self.kmeans_models.cluster_map_)
        self.assertEqual(sil.optimal_num_clusters_, 2)

#
# OtterTune - test_analysis.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.test import TestCase
from sklearn import datasets
from analysis.cluster import KMeansClusters, create_kselection_model


class TestCluster(TestCase):

    def test_cluster(self):

        # Load Iris data
        iris = datasets.load_iris()
        matrix = iris.data
        labels = iris.target
        kmeans_models = KMeansClusters()
        kmeans_models.fit(matrix, min_cluster=1,
                          max_cluster=20,
                          sample_labels=labels,
                          estimator_params={'n_init': 50})

        # Compute optimal # cluster using det-k
        detk = create_kselection_model("det-k")
        detk.fit(matrix, kmeans_models.cluster_map_)
        self.assertGreaterEqual(detk.optimal_num_clusters_, 1)
        self.assertLessEqual(detk.optimal_num_clusters_, 20)

        # Compute optimal # cluster using gap-statistics
        gapk = create_kselection_model("gap-statistic")
        gapk.fit(matrix, kmeans_models.cluster_map_)
        self.assertGreaterEqual(gapk.optimal_num_clusters_, 1)
        self.assertLessEqual(gapk.optimal_num_clusters_, 20)

        # Compute optimal # cluster using Silhouette Analysis
        sil_k = create_kselection_model("s-score")
        sil_k.fit(matrix, kmeans_models.cluster_map_)
        self.assertGreaterEqual(gapk.optimal_num_clusters_, 1)
        self.assertLessEqual(gapk.optimal_num_clusters_, 20)

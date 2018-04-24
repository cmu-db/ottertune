#
# OtterTune - test_preprocessing.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import unittest
import numpy as np

from analysis.preprocessing import DummyEncoder, consolidate_columnlabels


class TestDummyEncoder(unittest.TestCase):

    def test_no_categoricals(self):
        X = [[1, 2, 3], [4, 5, 6]]
        n_values = []
        categorical_features = []
        cat_columnlabels = []
        noncat_columnlabels = ['a', 'b', 'c']

        enc = DummyEncoder(n_values, categorical_features,
                           cat_columnlabels, noncat_columnlabels)
        X_encoded = enc.fit_transform(X)
        new_labels = enc.new_labels
        self.assertTrue(np.all(X == X_encoded))
        self.assertEqual(noncat_columnlabels, new_labels)

    def test_simple_categorical(self):
        X = [[0, 1, 2], [1, 1, 2], [2, 1, 2]]
        n_values = [3]
        categorical_features = [0]
        cat_columnlabels = ['label']
        noncat_columnlabels = ['a', 'b']

        X_expected = [[1, 0, 0, 1, 2], [0, 1, 0, 1, 2], [0, 0, 1, 1, 2]]
        new_labels_expected = ['label____0', 'label____1', 'label____2', 'a', 'b']
        enc = DummyEncoder(n_values, categorical_features,
                           cat_columnlabels, noncat_columnlabels)
        X_encoded = enc.fit_transform(X)
        new_labels = enc.new_labels
        self.assertTrue(np.all(X_expected == X_encoded))
        self.assertEqual(new_labels_expected, new_labels)

    def test_mixed_categorical(self):
        X = [[1, 0, 2], [1, 1, 2], [1, 2, 2]]
        n_values = [3]
        categorical_features = [1]
        cat_columnlabels = ['label']
        noncat_columnlabels = ['a', 'b']

        X_expected = [[1, 0, 0, 1, 2], [0, 1, 0, 1, 2], [0, 0, 1, 1, 2]]
        new_labels_expected = ['label____0', 'label____1', 'label____2', 'a', 'b']
        enc = DummyEncoder(n_values, categorical_features,
                           cat_columnlabels, noncat_columnlabels)
        X_encoded = enc.fit_transform(X)
        new_labels = enc.new_labels
        self.assertTrue(np.all(X_expected == X_encoded))
        self.assertEqual(new_labels_expected, new_labels)

    def test_consolidate(self):
        labels = ['label1____0', 'label1____1', 'label2____0', 'label2____1', 'noncat']
        consolidated = consolidate_columnlabels(labels)
        expected = ['label1', 'label2', 'noncat']
        self.assertEqual(expected, consolidated)

    def test_inverse_transform(self):
        X = [[1, 0, 2], [1, 1, 2], [1, 2, 2]]
        n_values = [3]
        categorical_features = [1]
        cat_columnlabels = ['label']
        noncat_columnlabels = ['a', 'b']

        X_expected = [[1, 0, 0, 1, 2], [0, 1, 0, 1, 2], [0, 0, 1, 1, 2]]
        enc = DummyEncoder(n_values, categorical_features,
                           cat_columnlabels, noncat_columnlabels)
        X_encoded = enc.fit_transform(X)
        self.assertTrue(np.all(X_encoded == X_expected))
        X_decoded = enc.inverse_transform(X_encoded)
        self.assertTrue(np.all(X == X_decoded))


if __name__ == '__main__':
    unittest.main()

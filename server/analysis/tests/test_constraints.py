#
# OtterTune - test_constraints.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import unittest
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from analysis.constraints import ParamConstraintHelper
from analysis.preprocessing import DummyEncoder


class ConstraintHelperTestCase(unittest.TestCase):

    def test_scale_rescale(self):
        X = datasets.load_boston()['data']
        X_scaler = StandardScaler()
        # params hard-coded for test (messy to import constant from website module)
        constraint_helper = ParamConstraintHelper(X_scaler, None,
                                                  init_flip_prob=0.3,
                                                  flip_prob_decay=0.5)
        X_scaled = X_scaler.fit_transform(X)
        # there may be some floating point imprecision between scaling and rescaling
        row_unscaled = np.round(constraint_helper._handle_scaling(X_scaled[0], True), 10)  # pylint: disable=protected-access
        self.assertTrue(np.all(X[0] == row_unscaled))
        row_rescaled = constraint_helper._handle_rescaling(row_unscaled, True)  # pylint: disable=protected-access
        self.assertTrue(np.all(X_scaled[0] == row_rescaled))

    def test_apply_constraints_unscaled(self):
        n_values = [3]
        categorical_features = [0]
        encoder = DummyEncoder(n_values, categorical_features, ['a'], [])
        encoder.fit([[0, 17]])
        X_scaler = StandardScaler()
        constraint_helper = ParamConstraintHelper(X_scaler, encoder,
                                                  init_flip_prob=0.3,
                                                  flip_prob_decay=0.5)

        X = [0.1, 0.2, 0.3, 17]
        X_expected = [0, 0, 1, 17]
        X_corrected = constraint_helper.apply_constraints(X, scaled=False, rescale=False)
        self.assertTrue(np.all(X_corrected == X_expected))

    def test_apply_constraints(self):
        n_values = [3]
        categorical_features = [0]
        encoder = DummyEncoder(n_values, categorical_features, ['a'], [])
        encoder.fit([[0, 17]])
        X_scaler = StandardScaler()
        constraint_helper = ParamConstraintHelper(X_scaler, encoder,
                                                  init_flip_prob=0.3,
                                                  flip_prob_decay=0.5)

        X = np.array([[0, 0, 1, 17], [1, 0, 0, 17]], dtype=float)
        X_scaled = X_scaler.fit_transform(X)
        row = X_scaled[0]
        new_row = np.copy(row)
        new_row[0: 3] += 0.1  # should still represent [0, 0, 1] encoding
        row_corrected = constraint_helper.apply_constraints(new_row)
        self.assertTrue(np.all(row == row_corrected))

    def test_randomize_categorical_features(self):
        n_values = [3]
        categorical_features = [0]
        encoder = DummyEncoder(n_values, categorical_features, ['a'], [])
        encoder.fit([[0, 17]])
        X_scaler = StandardScaler()
        constraint_helper = ParamConstraintHelper(X_scaler, encoder,
                                                  init_flip_prob=0.3,
                                                  flip_prob_decay=0.5)

        row = np.array([0, 0, 1, 17], dtype=float)
        counts = [0, 0, 0]
        trials = 20
        for _ in range(trials):
            row = constraint_helper.randomize_categorical_features(row, scaled=False, rescale=False)
            dummies = row[0: 3]
            self.assertTrue(np.all(np.logical_or(dummies == 0, dummies == 1)))
            self.assertEqual(np.sum(dummies), 1)
            counts[np.argmax(dummies)] += 1

        # this part of the test is non-deterministic, but I think failure
        # is a sign that this approach is not sufficiently random
        for ct in counts:
            self.assertTrue(ct > 0)


if __name__ == '__main__':
    unittest.main()

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
        X = np.array([[0, 0, 1, 17], [1, 0, 0, 17]], dtype=float)
        X_scaled = X_scaler.fit_transform(X)
        constraint_helper = ParamConstraintHelper(X_scaler, encoder,
                                                  init_flip_prob=0.3,
                                                  flip_prob_decay=0.5)

        row = X_scaled[0]
        new_row = np.copy(row)
        new_row[0: 3] += 0.1  # should still represent [0, 0, 1] encoding
        row_corrected = constraint_helper.apply_constraints(new_row)
        self.assertTrue(np.all(row == row_corrected))

    # tests that repeatedly applying randomize_categorical_features
    # always results in valid configurations of categorical dumny encodings
    # and will lead to all possible values of categorical variables being tried
    def test_randomize_categorical_features(self):
        # variable 0 is categorical, 3 values
        # variable 1 is not categorical
        # variable 2 is categorical, 4 values
        cat_var_0_levels = 3
        cat_var_2_levels = 4
        cat_var_0_idx = 0
        cat_var_2_idx = 2
        n_values = [cat_var_0_levels, cat_var_2_levels]
        categorical_features = [cat_var_0_idx, cat_var_2_idx]
        encoder = DummyEncoder(n_values, categorical_features, ['a', 'b'], [])
        encoder.fit([[0, 17, 0]])
        X_scaler = StandardScaler()
        constraint_helper = ParamConstraintHelper(X_scaler, encoder,
                                                  init_flip_prob=0.3,
                                                  flip_prob_decay=0.5)

        # row is a sample encoded set of features,
        # note that the non-categorical variable is on the right
        row = np.array([0, 0, 1, 1, 0, 0, 0, 17], dtype=float)
        trials = 20
        cat_var_0_counts = np.zeros(cat_var_0_levels)
        cat_var_2_counts = np.zeros(cat_var_2_levels)
        for _ in range(trials):
            # possibly flip the categorical features
            row = constraint_helper.randomize_categorical_features(row, scaled=False, rescale=False)

            # check that result is valid for cat_var_0
            cat_var_0_dummies = row[0: cat_var_0_levels]
            self.assertTrue(np.all(np.logical_or(cat_var_0_dummies == 0, cat_var_0_dummies == 1)))
            self.assertEqual(np.sum(cat_var_0_dummies), 1)
            cat_var_0_counts[np.argmax(cat_var_0_dummies)] += 1

            # check that result is valid for cat_var_2
            cat_var_2_dummies = row[cat_var_0_levels: cat_var_0_levels + cat_var_2_levels]
            self.assertTrue(np.all(np.logical_or(cat_var_2_dummies == 0, cat_var_2_dummies == 1)))
            self.assertEqual(np.sum(cat_var_2_dummies), 1)
            cat_var_2_counts[np.argmax(cat_var_2_dummies)] += 1

            self.assertEqual(row[-1], 17)

        for ct in cat_var_0_counts:
            self.assertTrue(ct > 0)

        for ct in cat_var_2_counts:
            self.assertTrue(ct > 0)


if __name__ == '__main__':
    unittest.main()

#
# OtterTune - lasso.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Jul 8, 2016

@author: dvanaken
'''

import numpy as np
from sklearn.linear_model import lasso_path

from .base import ModelBase


class LassoPath(ModelBase):
    """Lasso:

    Computes the Lasso path using Sklearn's lasso_path method.


    See also
    --------
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html


    Attributes
    ----------
    feature_labels_ : array, [n_features]
                      Labels for each of the features in X.

    alphas_ : array, [n_alphas]
              The alphas along the path where models are computed. (These are
              the decreasing values of the penalty along the path).

    coefs_ : array, [n_outputs, n_features, n_alphas]
             Coefficients along the path.

    rankings_ : array, [n_features]
             The average ranking of each feature across all target values.
    """
    def __init__(self):
        self.feature_labels_ = None
        self.alphas_ = None
        self.coefs_ = None
        self.rankings_ = None

    def _reset(self):
        """Resets all attributes (erases the model)"""
        self.feature_labels_ = None
        self.alphas_ = None
        self.coefs_ = None
        self.rankings_ = None

    def fit(self, X, y, feature_labels, estimator_params=None):
        """Computes the Lasso path using Sklearn's lasso_path method.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data (the independent variables).

        y : array-like, shape (n_samples, n_outputs)
            Training data (the output/target values).

        feature_labels : array-like, shape (n_features)
                         Labels for each of the features in X.

        estimator_params : dict, optional
                           The parameters to pass to Sklearn's Lasso estimator.


        Returns
        -------
        self
        """
        self._reset()
        if estimator_params is None:
            estimator_params = {}
        self.feature_labels_ = feature_labels

        alphas, coefs, _ = lasso_path(X, y, **estimator_params)
        self.alphas_ = alphas.copy()
        self.coefs_ = coefs.copy()

        # Rank the features in X by order of importance. This ranking is based
        # on how early a given features enter the regression (the earlier a
        # feature enters the regression, the MORE important it is).
        feature_rankings = [[] for _ in range(X.shape[1])]
        for target_coef_paths in self.coefs_:
            for i, feature_path in enumerate(target_coef_paths):
                entrance_step = 1
                for val_at_step in feature_path:
                    if val_at_step == 0:
                        entrance_step += 1
                    else:
                        break
                feature_rankings[i].append(entrance_step)
        self.rankings_ = np.array([np.mean(ranks) for ranks in feature_rankings])
        return self

    def get_ranked_features(self):
        if self.rankings_ is None:
            raise Exception("No lasso path has been fit yet!")

        rank_idxs = np.argsort(self.rankings_)
        return [self.feature_labels_[i] for i in rank_idxs]

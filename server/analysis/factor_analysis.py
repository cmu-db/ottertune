#
# OtterTune - factor_analysis.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Jul 4, 2016

@author: dvanaken
'''

import numpy as np
from sklearn.decomposition import FactorAnalysis as SklearnFactorAnalysis

from .base import ModelBase


class FactorAnalysis(ModelBase):
    """FactorAnalysis (FA):

    Fits an Sklearn FactorAnalysis model to X.


    See also
    --------
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html


    Attributes
    ----------
    model_ : sklearn.decomposition.FactorAnalysis
             The fitted FA model

    components_ : array, [n_components, n_features]
                  Components (i.e., factors) with maximum variance

    feature_labels_ : array, [n_features]

    total_variance_ : float
                      The total amount of variance explained by the components

    pvars_ : array, [n_components]
             The percentage of the variance explained by each component

    pvars_noise_ : array, [n_components]
                   The percentage of the variance explained by each component also
                   accounting for noise
    """

    def __init__(self):
        self.model_ = None
        self.components_ = None
        self.feature_labels_ = None
        self.total_variance_ = None
        self.pvars_ = None
        self.pvars_noise_ = None

    def _reset(self):
        """Resets all attributes (erases the model)"""
        self.model_ = None
        self.components_ = None
        self.feature_labels_ = None
        self.total_variance_ = None
        self.pvars_ = None
        self.pvars_noise_ = None

    def fit(self, X, feature_labels=None, n_components=None, estimator_params=None):
        """Fits an Sklearn FA model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        feature_labels : array-like, shape (n_features), optional
                         Labels for each of the features in X.

        estimator_params : dict, optional
                           The parameters to pass to Sklearn's FA estimators.


        Returns
        -------
        self
        """
        self._reset()
        if feature_labels is None:
            feature_labels = ["feature_{}".format(i) for i in range(X.shape[1])]
        self.feature_labels_ = feature_labels
        if n_components is not None:
            model = SklearnFactorAnalysis(n_components=n_components)
        else:
            model = SklearnFactorAnalysis()
        self.model_ = model
        if estimator_params is not None:
            # Update Sklearn estimator params
            assert isinstance(estimator_params, dict)
            self.model_.set_params(**estimator_params)
        self.model_.fit(X)

        # Remove zero-valued components (n_components x n_features)
        components_mask = np.sum(self.model_.components_ != 0.0, axis=1) > 0.0
        self.components_ = self.model_.components_[components_mask]

        # Compute the % variance explained (with/without noise)
        c2 = np.sum(self.components_ ** 2, axis=1)
        self.total_variance_ = np.sum(c2)
        self.pvars_ = 100 * c2 / self.total_variance_
        self.pvars_noise_ = 100 * c2 / (self.total_variance_ +
                                        np.sum(self.model_.noise_variance_))
        return self

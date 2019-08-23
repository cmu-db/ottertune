#
# OtterTune - gp.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Feb 18, 2018

@author: Bohan Zhang
'''
import numpy as np
from scipy.spatial.distance import cdist as ed
from scipy import special
from analysis.gp_tf import GPRResult


# numpy version of Gaussian Process Regression, not using Tensorflow
class GPRNP(object):

    def __init__(self, length_scale=1.0, magnitude=1.0, max_train_size=7000,
                 batch_size=3000, check_numerics=True, debug=False):
        assert np.isscalar(length_scale)
        assert np.isscalar(magnitude)
        assert length_scale > 0 and magnitude > 0
        self.length_scale = length_scale
        self.magnitude = magnitude
        self.max_train_size_ = max_train_size
        self.batch_size_ = batch_size
        self.check_numerics = check_numerics
        self.debug = debug
        self.X_train = None
        self.y_train = None
        self.K = None
        self.K_inv = None
        self.y_best = None

    def __repr__(self):
        rep = ""
        for k, v in sorted(self.__dict__.items()):
            rep += "{} = {}\n".format(k, v)
        return rep

    def __str__(self):
        return self.__repr__()

    def _reset(self):
        self.X_train = None
        self.y_train = None
        self.K = None
        self.K_inv = None
        self.y_best = None

    def check_X_y(self, X, y):
        from sklearn.utils.validation import check_X_y

        if X.shape[0] > self.max_train_size_:
            raise Exception("X_train size cannot exceed {} ({})"
                            .format(self.max_train_size_, X.shape[0]))
        return check_X_y(X, y, multi_output=True,
                         allow_nd=True, y_numeric=True,
                         estimator="GPRNP")

    def check_fitted(self):
        if self.X_train is None or self.y_train is None \
                or self.K is None:
            raise Exception("The model must be trained before making predictions!")

    @staticmethod
    def check_array(X):
        from sklearn.utils.validation import check_array
        return check_array(X, allow_nd=True, estimator="GPRNP")

    @staticmethod
    def check_output(X):
        finite_els = np.isfinite(X)
        if not np.all(finite_els):
            raise Exception("Input contains non-finite values: {}"
                            .format(X[~finite_els]))

    def fit(self, X_train, y_train, ridge=0.01):
        self._reset()
        X_train, y_train = self.check_X_y(X_train, y_train)
        if X_train.ndim != 2 or y_train.ndim != 2:
            raise Exception("X_train or y_train should have 2 dimensions! X_dim:{}, y_dim:{}"
                            .format(X_train.ndim, y_train.ndim))
        self.X_train = np.float32(X_train)
        self.y_train = np.float32(y_train)
        sample_size = self.X_train.shape[0]
        if np.isscalar(ridge):
            ridge = np.ones(sample_size) * ridge
        assert isinstance(ridge, np.ndarray)
        assert ridge.ndim == 1
        K = self.magnitude * np.exp(-ed(self.X_train, self.X_train) / self.length_scale) \
            + np.diag(ridge)
        K_inv = np.linalg.inv(K)
        self.K = K
        self.K_inv = K_inv
        self.y_best = np.min(y_train)
        return self

    def predict(self, X_test):
        self.check_fitted()
        if X_test.ndim != 2:
            raise Exception("X_test should have 2 dimensions! X_dim:{}"
                            .format(X_test.ndim))
        X_test = np.float32(GPRNP.check_array(X_test))
        test_size = X_test.shape[0]
        arr_offset = 0
        length_scale = self.length_scale
        yhats = np.zeros([test_size, 1])
        sigmas = np.zeros([test_size, 1])
        eips = np.zeros([test_size, 1])
        while arr_offset < test_size:
            if arr_offset + self.batch_size_ > test_size:
                end_offset = test_size
            else:
                end_offset = arr_offset + self.batch_size_
            xt_ = X_test[arr_offset:end_offset]
            K2 = self.magnitude * np.exp(-ed(self.X_train, xt_) / length_scale)
            K3 = self.magnitude * np.exp(-ed(xt_, xt_) / length_scale)
            K2_trans = np.transpose(K2)
            yhat = np.matmul(K2_trans, np.matmul(self.K_inv, self.y_train))
            sigma = np.sqrt(np.diag(K3 - np.matmul(K2_trans, np.matmul(self.K_inv, K2)))) \
                .reshape(xt_.shape[0], 1)
            u = (self.y_best - yhat) / sigma
            phi1 = 0.5 * special.erf(u / np.sqrt(2.0)) + 0.5
            phi2 = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(np.square(u) * (-0.5))
            eip = sigma * (u * phi1 + phi2)
            yhats[arr_offset:end_offset] = yhat
            sigmas[arr_offset:end_offset] = sigma
            eips[arr_offset:end_offset] = eip
            arr_offset = end_offset
        GPRNP.check_output(yhats)
        GPRNP.check_output(sigmas)
        return GPRResult(yhats, sigmas)

    def get_params(self, deep=True):
        return {"length_scale": self.length_scale,
                "magnitude": self.magnitude,
                "X_train": self.X_train,
                "y_train": self.y_train,
                "K": self.K,
                "K_inv": self.K_inv}

    def set_params(self, **parameters):
        for param, val in list(parameters.items()):
            setattr(self, param, val)
        return self

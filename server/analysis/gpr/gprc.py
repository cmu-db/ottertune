#
# OtterTune - analysis/gprc.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
# Author: Dana Van Aken

from __future__ import absolute_import

import tensorflow as tf
from gpflow import settings
from gpflow.decors import autoflow, name_scope, params_as_tensors
from gpflow.models import GPR


class GPRC(GPR):

    def __init__(self, X, Y, kern, mean_function=None, **kwargs):
        super(GPRC, self).__init__(X, Y, kern, mean_function, **kwargs)
        self.cholesky = None
        self.alpha = None

    @autoflow()
    def _compute_cache(self):
        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        L = tf.cholesky(K, name='gp_cholesky')
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X), name='gp_alpha')
        return L, V

    def update_cache(self):
        self.cholesky, self.alpha = self._compute_cache()

    @name_scope('predict')
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        if self.cholesky is None:
            self.update_cache()
        Kx = self.kern.K(self.X, Xnew)
        A = tf.matrix_triangular_solve(self.cholesky, Kx, lower=True)
        fmean = tf.matmul(A, self.alpha, transpose_a=True) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar

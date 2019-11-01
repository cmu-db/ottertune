#
# OtterTune - analysis/optimize.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
# Author: Dana Van Aken

import numpy as np
import tensorflow as tf
from gpflow import settings
from sklearn.utils import assert_all_finite, check_array
from sklearn.utils.validation import FLOAT_DTYPES

from analysis.util import get_analysis_logger

LOG = get_analysis_logger(__name__)


def tf_optimize(model, Xnew_arr, learning_rate=0.01, maxiter=100, ucb_beta=3.,
                active_dims=None, bounds=None):
    Xnew_arr = check_array(Xnew_arr, copy=False, warn_on_dtype=True, dtype=FLOAT_DTYPES)

    Xnew = tf.Variable(Xnew_arr, name='Xnew', dtype=settings.float_type)
    if bounds is None:
        lower_bound = tf.constant(-np.infty, dtype=settings.float_type)
        upper_bound = tf.constant(np.infty, dtype=settings.float_type)
    else:
        lower_bound = tf.constant(bounds[0], dtype=settings.float_type)
        upper_bound = tf.constant(bounds[1], dtype=settings.float_type)
    Xnew_bounded = tf.minimum(tf.maximum(Xnew, lower_bound), upper_bound)

    if active_dims:
        indices = []
        updates = []
        n_rows = Xnew_arr.shape[0]
        for c in active_dims:
            for r in range(n_rows):
                indices.append([r, c])
                updates.append(Xnew_bounded[r, c])
        part_X = tf.scatter_nd(indices, updates, Xnew_arr.shape)
        Xin = part_X + tf.stop_gradient(-part_X + Xnew_bounded)
    else:
        Xin = Xnew_bounded

    beta_t = tf.constant(ucb_beta, name='ucb_beta', dtype=settings.float_type)
    y_mean_var = model.likelihood.predict_mean_and_var(*model._build_predict(Xin))
    loss = tf.subtract(y_mean_var[0], tf.multiply(beta_t, y_mean_var[1]), name='loss_fn')
    opt = tf.train.AdamOptimizer(learning_rate)
    train_op = opt.minimize(loss)
    variables = opt.variables()
    init_op = tf.variables_initializer([Xnew] + variables)
    session = model.enquire_session(session=None)
    with session.as_default():
        session.run(init_op)
        for i in range(maxiter):
            session.run(train_op)
        Xnew_value = session.run(Xnew_bounded)
        y_mean_value, y_var_value = session.run(y_mean_var)
        loss_value = session.run(loss)
        assert_all_finite(Xnew_value)
        assert_all_finite(y_mean_value)
        assert_all_finite(y_var_value)
        assert_all_finite(loss_value)
        return Xnew_value, y_mean_value, y_var_value, loss_value

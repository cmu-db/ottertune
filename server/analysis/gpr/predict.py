#
# OtterTune - analysis/optimize.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
# Author: Dana Van Aken

import tensorflow as tf
from sklearn.utils import assert_all_finite, check_array
from sklearn.utils.validation import FLOAT_DTYPES


class GPRResult():

    def __init__(self, ypreds=None, sigmas=None):
        self.ypreds = ypreds
        self.sigmas = sigmas


def gpflow_predict(model, Xin):
    Xin = check_array(Xin, copy=False, warn_on_dtype=True, dtype=FLOAT_DTYPES)
    fmean, fvar, _, _, _ = model._build_predict(Xin)  # pylint: disable=protected-access
    y_mean_var = model.likelihood.predict_mean_and_var(fmean, fvar)
    y_mean = y_mean_var[0]
    y_var = y_mean_var[1]
    y_std = tf.sqrt(y_var)

    session = model.enquire_session(session=None)
    with session.as_default():
        y_mean_value = session.run(y_mean)
        y_std_value = session.run(y_std)
        assert_all_finite(y_mean_value)
        assert_all_finite(y_std_value)
        return GPRResult(y_mean_value, y_std_value)

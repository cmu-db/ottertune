#
# OtterTune - util.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Oct 24, 2017

@author: dva
'''

import logging
from numbers import Number

import contextlib
import datetime
import numpy as np


def get_analysis_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(
        fmt='%(asctime)s [%(funcName)s:%(lineno)03d] %(levelname)-5s: %(message)s',
        datefmt='%m-%d-%Y %H:%M:%S'
    )
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    logger.setLevel(level)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    return logger


LOG = get_analysis_logger(__name__)


def stdev_zero(data, axis=None, nearzero=1e-8):
    mstd = np.expand_dims(data.std(axis=axis), axis=axis)
    return (np.abs(mstd) < nearzero).squeeze()


def get_datetime():
    return datetime.datetime.utcnow()


class TimerStruct(object):

    def __init__(self):
        self.__start_time = 0.0
        self.__stop_time = 0.0
        self.__elapsed = None

    @property
    def elapsed_seconds(self):
        if self.__elapsed is None:
            return (get_datetime() - self.__start_time).total_seconds()
        return self.__elapsed.total_seconds()

    def start(self):
        self.__start_time = get_datetime()

    def stop(self):
        self.__stop_time = get_datetime()
        self.__elapsed = (self.__stop_time - self.__start_time)


@contextlib.contextmanager
def stopwatch(message=None):
    ts = TimerStruct()
    ts.start()
    try:
        yield ts
    finally:
        ts.stop()
        if message is not None:
            LOG.info('Total elapsed_seconds time for %s: %.3fs', message, ts.elapsed_seconds)


def get_data_base(arr):
    """For a given Numpy array, finds the
    base array that "owns" the actual data."""
    base = arr
    while isinstance(base.base, np.ndarray):
        base = base.base
    return base


def arrays_share_data(x, y):
    return get_data_base(x) is get_data_base(y)


def array_tostring(arr):
    arr_shape = arr.shape
    arr = arr.ravel()
    arr = np.array([str(a) for a in arr])
    return arr.reshape(arr_shape)


def is_numeric_matrix(matrix):
    assert matrix.size > 0
    return isinstance(matrix.ravel()[0], Number)


def is_lexical_matrix(matrix):
    assert matrix.size > 0
    return isinstance(matrix.ravel()[0], str)

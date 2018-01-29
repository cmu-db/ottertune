#
# OtterTune - cross_validation.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Aug 29, 2016

@author: dvanaken
'''

from collections import namedtuple

import cPickle as pickle
import gc
import numpy as np
import tensorflow as tf
from fabric.api import local
from sklearn.model_selection import KFold
from sklearn.utils.validation import _is_arraylike, check_X_y

from .util import get_analysis_logger, stopwatch

LOG = get_analysis_logger(__name__)

GridScore = namedtuple('GridScore', ['parameters', 'mean_scores', 'cv_scores'])


def mp_grid_search((task_id,     # pylint: disable=unused-argument
                    parameters,
                    estimator,
                    kf,
                    X,
                    y,
                    score_fns,
                    ntasks)):    # pylint: disable=unused-argument
    estimator.set_params(**parameters)
    sparams = estimator.get_params()
    nfolds = kf.n_folds
    scores = np.empty((nfolds, len(score_fns)))

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        with tf.Graph().as_default():
            estimator.fit(X_train, y_train)
            ypreds, sigmas, _ = estimator.predict(X_test)
        gc.collect()
        for j, score_fn in enumerate(score_fns):
            scores[i, j] = score_fn(y_test, ypreds, sigmas)
        LOG.info("\tCompleted %d/%d folds", i + 1, nfolds)
    mean_scores = scores.mean(axis=0)
    grid_score = GridScore(sparams, mean_scores, scores)
    return grid_score


class GridSearch(object):

    def __init__(self, estimator_cls, parameter_grid, score_fns,
                 nfolds=10, shuffle=False, seed=None, njobs=1,
                 checkpoint_path=None):
        self.estimator_cls = estimator_cls
        self.parameter_grid = parameter_grid
        self.nfolds = nfolds
        self.seed = seed
        assert njobs == 1, "# jobs > 1 not supported."
        self.njobs = njobs
        assert _is_arraylike(score_fns)
        self.score_fns = score_fns
        self.checkpoint_path = checkpoint_path
        self.grid_scores = None
        self.kf = KFold(n_splits=self.nfolds,
                        shuffle=shuffle,
                        random_state=seed)

    def __repr__(self):
        rep = ""
        for k, v in self.__dict__.iteritems():
            rep += "{} = {}\n".format(k, v)
        return rep

    def __str__(self):
        return self.__repr__()

    def fit(self, X, y):
        X, y = check_X_y(X, y, allow_nd=True, multi_output=True,
                         y_numeric=True, estimator="GridSearch")
        LOG.info("njobs = %d", self.njobs)
        if self.njobs > 1:
            assert False
#             iterable = [(i, pg, self.estimator_cls, self.kf, X, y, \
#                          self.score_fns, len(self.parameter_grid)) \
#                          for i,pg in enumerate(self.parameter_grid)]
#             try:
#                 p = multiprocessing.Pool(self.njobs)
#                 res = p.map(mp_grid_search, iterable)
#             except:
#                 traceback.print_exc()
        else:
            self.grid_scores = []
            estimator = self.estimator_cls()
            num_tasks = len(self.parameter_grid)
            for i, params in enumerate(self.parameter_grid):
                LOG.info("Starting task %d/%d...", i + 1, num_tasks)
                with stopwatch("Done. Elapsed time"):
                    self.grid_scores.append(mp_grid_search((i,
                                                            params,
                                                            estimator,
                                                            self.kf,
                                                            X,
                                                            y,
                                                            self.score_fns,
                                                            len(self.parameter_grid))))

                if self.checkpoint_path is not None:
                    local("rm -f {}*.p".format(self.checkpoint_path))
                    savepath = self.checkpoint_path + "_{}.p".format(i)
                    with open(savepath, 'w') as f:
                        pickle.dump(self.grid_scores, f)

    @staticmethod
    def create_parameter_grid(param_dict):
        from sklearn.model_selection import ParameterGrid
        return ParameterGrid(param_dict)


def rmse_cv(y_reals, y_preds, sigmas):  # pylint: disable=unused-argument
    assert y_preds.shape == y_reals.shape
    return np.sqrt(np.mean(np.square(y_preds - y_reals)))


def gpvar_cv(y_reals, y_preds, sigmas):
    y_reals = y_reals.ravel()
    y_preds = y_preds.ravel()
    sigmas = sigmas.ravel()
    y_upper = y_preds + 1.96 * sigmas
    y_lower = y_preds - 1.96 * sigmas
    bounded_ys = np.logical_and(y_reals <= y_upper, y_reals >= y_lower)
    res = 1 - float(np.sum(bounded_ys)) / y_preds.shape[0]
    assert res >= 0 and res <= 1
    return res


CombinedScore = namedtuple('CombinedScore', ['combined_scores',
                                             'scaled_scores',
                                             'combined_indices',
                                             'rmse_indices',
                                             'gpvar_indices'])


def combine_rmse_gpvar(grid_scores, w_rmse=0.8, w_gpvar=0.2):
    from sklearn.preprocessing import minmax_scale

    # Scale rmses, gpvars to (0,1)
    scaled_scores = np.empty((len(grid_scores), 2))
    for i, scores in enumerate(grid_scores):
        scaled_scores[i, 0] = scores.mean_scores[0]
        scaled_scores[i, 1] = scores.mean_scores[1]
    rmse_sort_indices = np.argsort(scaled_scores[:, 0])
    gpvar_sort_indices = np.argsort(scaled_scores[:, 1])
    scaled_scores = minmax_scale(scaled_scores)
    combined_scores = w_rmse * scaled_scores[:, 0] + w_gpvar * scaled_scores[:, 1]
    comb_sort_indices = np.argsort(combined_scores)
    return CombinedScore(combined_scores,
                         scaled_scores,
                         comb_sort_indices,
                         rmse_sort_indices,
                         gpvar_sort_indices)

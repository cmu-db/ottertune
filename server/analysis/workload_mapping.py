#
# OtterTune - workload_mapping.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Jul 11, 2016

@author: dvanaken
'''

import copy
import gc
import multiprocessing
import operator
import os
import warnings
import zlib
import cPickle as pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from . import preprocessing as prep
from .gp_tf import GPR
from .matrix import Matrix
from .util import get_analysis_logger, stopwatch

warnings.simplefilter(action="ignore", category=FutureWarning)

LOG = get_analysis_logger(__name__)


class WorkloadState(object):

    def __init__(self, X=None, y=None, models=None):
        self.X = X
        self.y = y
        self.models = models

    @staticmethod
    def compress(workload_state):
        return zlib.compress(pickle.dumps(workload_state))

    @staticmethod
    def decompress(compressed_workload_state):
        return pickle.loads(zlib.decompress(compressed_workload_state))


def worker_create_model((worker_id, workload_name, data, njobs)):
    LOG.info("%d: building models for %s", worker_id, os.path.basename(workload_name))
    X, y = data
    models = {}
    for col, label in zip(y.data.T, y.columnlabels):
        LOG.debug("    %d: %s", worker_id, label)
        length_scale, magnitude, ridge_const = 1., 1., 1.
        ridge = np.ones(X.data.shape[0]) * ridge_const
        col = col.reshape(-1, 1)
        model = GPR(length_scale, magnitude)
        model.fit(X.data, col, ridge)
        models[label] = model
    workload_state = WorkloadState(X, y, models)
    workload_state = WorkloadState.compress(workload_state)
    LOG.debug("%d: done. (%d/%d)", worker_id, worker_id + 1, njobs)
    return (workload_name, workload_state)


def worker_score_workload((worker_id, workload_name, workload_state,
                           X_client, y_client, njobs)):
    LOG.debug("%d: computing scores for %s", worker_id, os.path.basename(workload_name))
    workload_state = WorkloadState.decompress(workload_state)
    assert np.array_equal(workload_state.X.columnlabels,
                          X_client.columnlabels)
    assert np.array_equal(workload_state.y.columnlabels,
                          y_client.columnlabels)

    # Make all predictions
    metrics = workload_state.y.columnlabels
    predictions = np.empty_like(y_client.data)
    for i, metric in enumerate(metrics):
        LOG.debug("    %d: %s", worker_id, metric)
        model = workload_state.models[metric]
        res = model.predict(X_client.data)
        predictions[:, i] = res.ypreds.ravel()

    # Compute distance
    dists = np.sum(np.square(np.subtract(predictions, y_client.data)), axis=1)
    assert dists.shape == (predictions.shape[0],)
    LOG.debug("%d: done. (%d/%d)", worker_id, worker_id + 1, njobs)
    return (workload_name, np.mean(dists))


class WorkloadMapper(object):

    POOL_SIZE = 8
    MAX_SAMPLES = 5000

    def __init__(self, dbms_name, featured_knobs, featured_metrics,
                 target_workload_name, workload_repo_dirs):
        self.workload_states_ = None
        self.dbms_name_ = dbms_name

#         workload_dirs = glob.glob(os.path.join(Paths.DATADIR,
#                                                "analysis*{}*{}*".format(dbms,
#                                                                         cluster)))
#         if tuner.incremental_knob_selection:
#             self.featured_knobs_ = tuner.get_n_featured_knobs(tuner.max_knobs)
#         else:
        self.featured_knobs_ = featured_knobs
        self.featured_metrics_ = featured_metrics

#         target_wkld_desc = exp.exp_id(exp.benchmark)
        self.workload_dirs_ = [w for w in workload_repo_dirs if not
                               w.endswith(target_workload_name)]
        assert len(self.workload_dirs_) > 0

        pool_size = min(len(self.workload_dirs_), self.POOL_SIZE)
        if pool_size > 1:
            self.pool_ = multiprocessing.Pool(pool_size)
        else:
            self.pool_ = None
        self.initialize_models()
        assert self.workload_states_ is not None
        gc.collect()

    def initialize_models(self):
        LOG.info("Initializing models for # knobs=%d\n", self.featured_knobs_.size)
        with stopwatch("workload mapping model creation"):
            n_values, cat_indices, params = prep.dummy_encoder_helper(self.dbms_name,
                                                                      self.featured_knobs_)
            assert isinstance(n_values, np.ndarray)
            if n_values.size > 0:
                self.dummy_encoder_ = prep.DummyEncoder(n_values, cat_indices)
            else:
                self.dummy_encoder_ = None
            self.X_scaler_ = StandardScaler()
            self.y_scaler_ = StandardScaler()
            data_map = {}
            for i, wd in enumerate(self.workload_dirs_):
                # Load and filter data
                Xpath = os.path.join(wd, "X_data_enc.npz")
                ypath = os.path.join(wd, "y_data_enc.npz")
                X = Matrix.load_matrix(Xpath)
                y = Matrix.load_matrix(ypath)
                X = X.filter(self.featured_knobs_, "columns")
                y = y.filter(self.featured_metrics_, "columns")
                assert np.array_equal(X.columnlabels, self.featured_knobs_)
                assert np.array_equal(y.columnlabels, self.featured_metrics_)
                assert np.array_equal(X.rowlabels, y.rowlabels)
                num_samples = X.shape[0]
                if num_samples > self.MAX_SAMPLES:
                    LOG.info("Shrinking %d samples to %d", num_samples, self.MAX_SAMPLES)
                    rand_indices = prep.get_shuffle_indices(num_samples)[:self.MAX_SAMPLES]
                    X = Matrix(X.data[rand_indices],
                               X.rowlabels[rand_indices],
                               X.columnlabels)
                    y = Matrix(y.data[rand_indices],
                               y.rowlabels[rand_indices],
                               y.columnlabels)
                num_samples = X.shape[0]
                assert num_samples <= self.MAX_SAMPLES
                assert num_samples == y.shape[0]

                # Dummy-code categorical knobs
                if self.dummy_encoder_ is not None:
                    if i == 0:
                        # Just need to fit this once
                        self.dummy_encoder_.fit(X.data, columnlabels=X.columnlabels)
                    X = Matrix(self.dummy_encoder_.transform(X.data),
                               X.rowlabels,
                               self.dummy_encoder_.columnlabels)

                self.X_scaler_.partial_fit(X.data)
                self.y_scaler_.partial_fit(y.data)
                data_map[wd] = (X, y)

            if self.dummy_encoder_ is not None:
                # Fix X_scaler wrt categorical features
                prep.fix_scaler(self.X_scaler_, self.dummy_encoder_, params)

            # Scale X/y
            all_ys = []
            for wd, (X, y) in data_map.iteritems():
                X.data = self.X_scaler_.transform(X.data)
                y.data = self.y_scaler_.transform(y.data)
                all_ys.append(y.data)

            # Concat all ys and compute deciles
            all_ys = np.vstack(all_ys)
            self.y_binner_ = prep.Bin(0, axis=0)
            self.y_binner_.fit(all_ys)
            del all_ys

            # Bin y by deciles and fit scaler
            self.y_gp_scaler_ = StandardScaler()
            for wd, (X, y) in data_map.iteritems():
                y.data = self.y_binner_.transform(y.data)
                self.y_gp_scaler_.partial_fit(y.data)

            # Recenter y-values
            for wd, (X, y) in data_map.iteritems():
                y.data = self.y_gp_scaler_.transform(y.data)

            njobs = len(data_map)
            iterable = [(i, wd, ws, njobs) for i, (wd, ws)
                        in enumerate(data_map.iteritems())]
            if self.pool_ is not None:
                res = self.pool_.map(worker_create_model, iterable)
            else:
                res = []
                for item in iterable:
                    res.append(worker_create_model(item))
            self.workload_states_ = dict(res)

    def map_workload(self, X_client, y_client):
        with stopwatch("workload mapping - preprocessing"):
            # Filter be featured knobs & metrics
            X_client = X_client.filter(self.featured_knobs_, "columns")
            y_client = y_client.filter(self.featured_metrics_, "columns")

            # Generate unique X,y matrices
            X_client, y_client = Matrix.get_unique_matrix(X_client, y_client)

            # Preprocessing steps
            if self.dummy_encoder_ is not None:
                X_client = Matrix(self.dummy_encoder_.transform(X_client.data),
                                  X_client.rowlabels,
                                  self.dummy_encoder_.columnlabels)
            X_client.data = self.X_scaler_.transform(X_client.data)

            # Create y_client scaler with prior and transform client data
            y_client_scaler = copy.deepcopy(self.y_scaler_)
            y_client_scaler.n_samples_seen_ = 1
            y_client_scaler.partial_fit(y_client.data)
            y_client.data = y_client_scaler.transform(y_client.data)

            # Bin and recenter client data
            y_client.data = self.y_binner_.transform(y_client.data)
            y_client.data = self.y_gp_scaler_.transform(y_client.data)

            # Compute workload scores in parallel
            njobs = len(self.workload_states_)
            iterable = [(i, wd, ws, X_client, y_client, njobs)
                        for i, (wd, ws) in enumerate(self.workload_states_.iteritems())]

        with stopwatch("workload mapping - predictions"):
            if self.pool_ is not None:
                wkld_scores = self.pool_.map(worker_score_workload, iterable)
            else:
                wkld_scores = []
                for item in iterable:
                    wkld_scores.append(worker_score_workload(item))

        sorted_wkld_scores = sorted(wkld_scores, key=operator.itemgetter(1))

        LOG.info("WORKLOAD_SCORES")
        for wkld, score in sorted_wkld_scores:
            LOG.info("    %s: %0.2f", os.path.basename(wkld), score)

        return sorted_wkld_scores[0][0]

#
# OtterTune - constraints.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Sep 8, 2016

@author: dvanaken
'''

import numpy as np


class ParamConstraintHelper(object):

    def __init__(self, scaler, encoder=None, binary_vars=None,
                 init_flip_prob=0.3, flip_prob_decay=0.5):
        if 'inverse_transform' not in dir(scaler):
            raise Exception("Scaler object must provide function inverse_transform(X)")
        if 'transform' not in dir(scaler):
            raise Exception("Scaler object must provide function transform(X)")
        self.scaler_ = scaler
        if encoder is not None and len(encoder.n_values) > 0:
            self.is_dummy_encoded_ = True
            self.encoder_ = encoder.encoder
        else:
            self.is_dummy_encoded_ = False
        self.binary_vars_ = binary_vars
        self.init_flip_prob_ = init_flip_prob
        self.flip_prob_decay_ = flip_prob_decay

    def apply_constraints(self, sample, scaled=True, rescale=True):
        conv_sample = self._handle_scaling(sample, scaled)

        if self.is_dummy_encoded_:
            # apply categorical (ie enum var, >=3 values) constraints
            n_values = self.encoder_.n_values_
            cat_start_indices = self.encoder_.feature_indices_
            for i, nvals in enumerate(n_values):
                start_idx = cat_start_indices[i]
                cvals = conv_sample[start_idx: start_idx + nvals]
                cvals = np.array(np.arange(nvals) == np.argmax(cvals), dtype=float)
                assert np.sum(cvals) == 1
                conv_sample[start_idx: start_idx + nvals] = cvals

        # apply binary (0-1) constraints
        if self.binary_vars_ is not None:
            for i in self.binary_vars_:
                # round to closest
                if conv_sample[i] >= 0.5:
                    conv_sample[i] = 1
                else:
                    conv_sample[i] = 0

        conv_sample = self._handle_rescaling(conv_sample, rescale)
        return conv_sample

    def _handle_scaling(self, sample, scaled):
        if scaled:
            if sample.ndim == 1:
                sample = sample.reshape(1, -1)
            sample = self.scaler_.inverse_transform(sample).ravel()
        else:
            sample = np.array(sample)
        return sample

    def _handle_rescaling(self, sample, rescale):
        if rescale:
            if sample.ndim == 1:
                sample = sample.reshape(1, -1)
            return self.scaler_.transform(sample).ravel()
        return sample

    def randomize_categorical_features(self, sample, scaled=True, rescale=True):
        # If there are no categorical features, this function is a no-op.
        if not self.is_dummy_encoded_:
            return sample
        n_values = self.encoder_.n_values_
        cat_start_indices = self.encoder_.feature_indices_
        n_cat_feats = len(n_values)

        conv_sample = self._handle_scaling(sample, scaled)
        flips = np.zeros((n_cat_feats,), dtype=bool)

        # Always flip at least one categorical feature
        flips[0] = True

        # Flip the rest with decreasing probability
        p = self.init_flip_prob_
        for i in range(1, n_cat_feats):
            if np.random.rand() <= p:
                flips[i] = True
            p *= self.flip_prob_decay_

        flip_shuffle_indices = np.random.choice(np.arange(n_cat_feats),
                                                n_cat_feats,
                                                replace=False)
        flips = flips[flip_shuffle_indices]

        for i, nvals in enumerate(n_values):
            if flips[i]:
                start_idx = cat_start_indices[i]
                current_val = conv_sample[start_idx: start_idx + nvals]
                assert np.all(np.logical_or(current_val == 0, current_val == 1)), \
                    "categorical {0}: value not 0/1: {1}".format(i, current_val)
                choices = np.arange(nvals)[current_val != 1]
                assert choices.size == nvals - 1
                r = np.zeros(nvals)
                r[np.random.choice(choices)] = 1
                assert np.sum(r) == 1
                conv_sample[start_idx: start_idx + nvals] = r

        conv_sample = self._handle_rescaling(conv_sample, rescale)
        return conv_sample

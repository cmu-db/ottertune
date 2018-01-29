#
# OtterTune - sampling.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Sep 13, 2016

@author: dvanaken
'''

import numpy as np
from pyDOE import lhs
from scipy.stats.distributions import norm, uniform

NORMAL_DISTRIBUTION_TYPE = 0
UNIFORM_DISTRIBUTION_TYPE = 1


class LHSSampler(object):

    def __init__(self, n_samples, locs, scales, feat_names):
        n_feats = locs.size
        assert scales.size == n_feats
        assert n_samples > 0
        self.feat_names_ = feat_names.copy()
        self.samples_ = gen_samples(n_feats, n_samples,
                                    loc=locs, scale=scales,
                                    distribution_type=UNIFORM_DISTRIBUTION_TYPE)
        self.incomplete_samples_ = self.samples_.copy()

    def has_next_sample(self):
        return self.incomplete_samples_ is not None

    def get_next_sample(self):
        if self.incomplete_samples_ is not None:
            assert self.incomplete_samples_.size > 0
            split = np.vsplit(self.incomplete_samples_, [1])
            assert len(split) == 2
            next_sample = split[0]
            if split[1].size == 0:
                self.incomplete_samples_ = None
            else:
                self.incomplete_samples_ = split[1]
            return next_sample.ravel()
        else:
            return None

    def get_samples(self):
        return self.samples_.copy()

    def get_feat_names(self):
        return self.feat_names_.copy()

    @staticmethod
    def create_sampler(n_samples, featured_knobs, knob_catalog, sampling_type="lhs"):
        n_feats = len(featured_knobs)
        locs = np.empty(n_feats)
        scales = np.empty(n_feats)
        for i, fname in enumerate(featured_knobs):
            p = knob_catalog[fname]
            if p.iscategorical:
                true_vals = p.valid_values
                pmin, pmax = 0, len(true_vals) - 1e-6
            else:
                if p.true_range is None:
                    true_vals = p.true_values
                    assert true_vals is not None
                    pmin, pmax = true_vals[0], true_vals[-1]
                else:
                    pmin, pmax = p.true_range

                if p.unit == "bytes":
                    if pmin <= 0:
                        pmin = 1
                    pmin = np.ceil(np.log2(pmin))
                    pmax = np.floor(np.log2(pmax))

            locs[i] = pmin
            scales[i] = pmax - pmin

        return LHSSampler(n_samples, locs, scales, featured_knobs)


def gen_samples(n_feats, n_samples, criterion='m',
                loc=None, scale=None,
                distribution_type=UNIFORM_DISTRIBUTION_TYPE):
    s = lhs(n_feats, samples=n_samples, criterion=criterion)
    if loc is not None:
        assert scale is not None
        if np.isscalar(loc):
            assert np.isscalar(scale)
            s = gen_sample(loc, scale, s, distribution_type)
        else:
            assert isinstance(loc, np.ndarray)
            assert isinstance(scale, np.ndarray)
            for i in range(n_feats):
                s[:, i] = gen_sample(loc[i], scale[i], s[:, i], distribution_type)
    return s


def gen_sample(loc, scale, sample, distribution_type):
    if distribution_type == NORMAL_DISTRIBUTION_TYPE:
        return norm(loc=loc, scale=scale).ppf(sample)
    elif distribution_type == UNIFORM_DISTRIBUTION_TYPE:
        return uniform(loc=loc, scale=scale).ppf(sample)
    else:
        raise Exception("Invalid distribution type: {}"
                        .format(distribution_type))

#
# OtterTune - analysis/gpr_models.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
# Author: Dana Van Aken

import copy
import json
import os

import gpflow
import numpy as np
import tensorflow as tf

from .gprc import GPRC


class BaseModel(object):

    # Min/max bounds for the kernel lengthscales
    _LENGTHSCALE_BOUNDS = (0.1, 10.)

    # Keys for each kernel's hyperparameters
    _KERNEL_HP_KEYS = []

    # The key for the likelihood parameter
    _LIKELIHOOD_HP_KEY = 'GPRC/likelihood/variance'

    def __init__(self, X, y, hyperparameters=None, optimize_hyperparameters=False,
                 learning_rate=0.001, maxiter=5000, **kwargs):
        # Store model kwargs
        self._model_kwargs = {
            'hyperparameters': hyperparameters,
            'optimize_hyperparameters': optimize_hyperparameters,
            'learning_rate': learning_rate,
            'maxiter': maxiter,
        }

        # Store kernel kwargs
        kernel_kwargs = self._get_kernel_kwargs(X_dim=X.shape[1], **kwargs)
        if hyperparameters is not None:
            self._assign_kernel_hyperparams(hyperparameters, kernel_kwargs)
        self._kernel_kwargs = copy.deepcopy(kernel_kwargs)

        # Build the kernels and the model
        with gpflow.defer_build():
            k = self._build_kernel(kernel_kwargs, optimize_hyperparameters=optimize_hyperparameters, **kwargs)
            m = GPRC(X, y, kern=k)
            if hyperparameters is not None and self._LIKELIHOOD_HP_KEY in hyperparameters:
                m.likelihood.variance = hyperparameters[self._LIKELIHOOD_HP_KEY]
        m.compile()

        # If enabled, optimize the hyperparameters
        if optimize_hyperparameters:
            opt = gpflow.train.AdamOptimizer(learning_rate)
            opt.minimize(m, maxiter=maxiter)
        self._model = m

    def _get_kernel_kwargs(self, **kwargs):
        return []

    def _build_kernel(self, kernel_kwargs, **kwargs):
        return None

    def get_hyperparameters(self):
        return {k: float(v) if v.ndim == 0 else v.tolist()
                for k, v in self._model.read_values().items()}

    def get_model_parameters(self):
        return {
            'model_params': copy.deepcopy(self._model_kwargs),
            'kernel_params': copy.deepcopy(self._kernel_kwargs)
        }

    def _assign_kernel_hyperparams(self, hyperparams, kernel_kwargs):
        for i, kernel_keys in enumerate(self._KERNEL_HP_KEYS):
            for key in kernel_keys:
                if key in hyperparams:
                    argname = key.rsplit('/', 1)[-1]
                    kernel_kwargs[i][argname] = hyperparams[key]

    @staticmethod
    def load_hyperparameters(path, hp_idx=0):
        with open(path, 'r') as f:
            hyperparams = json.load(f)['hyperparameters']
        if isinstance(hyperparams, list):
            assert hp_idx >= 0, 'hp_idx: {} (expected >= 0)'.format(hp_idx)
            if hp_idx >= len(hyperparams):
                hp_idx = -1
            hyperparams = hyperparams[hp_idx]
        return hyperparams


class BasicGP(BaseModel):

    _KERNEL_HP_KEYS = [
        [
            'GPRC/kern/kernels/0/variance',
            'GPRC/kern/kernels/0/lengthscales',
        ],
        [
            'GPRC/kern/kernels/1/variance',
        ],
    ]

    def _get_kernel_kwargs(self, **kwargs):
        X_dim = kwargs.pop('X_dim')
        return [
            {
                'input_dim': X_dim,
                'ARD': True
            },
            {
                'input_dim': X_dim,
            },
        ]

    def _build_kernel(self, kernel_kwargs, **kwargs):
        k0 = gpflow.kernels.Exponential(**kernel_kwargs[0])
        k1 = gpflow.kernels.White(**kernel_kwargs[1])
        if kwargs.pop('optimize_hyperparameters'):
            k0.lengthscales.transform = gpflow.transforms.Logistic(
                *self._LENGTHSCALE_BOUNDS)
        k = k0 + k1
        return k


_MODEL_MAP = {
    'BasicGP': BasicGP,
}


def create_model(model_name, **kwargs):
    # Update tensorflow session settings to enable GPU sharing
    gpflow.settings.session.update(gpu_options=tf.GPUOptions(allow_growth=True))
    check_valid(model_name)
    return _MODEL_MAP[model_name](**kwargs)


def check_valid(model_name):
    if model_name not in _MODEL_MAP:
        raise ValueError('Invalid GPR model name: {}'.format(model_name))

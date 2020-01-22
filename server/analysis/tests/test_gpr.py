#
# OtterTune - test_gpr.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import unittest
import random
import numpy as np
import gpflow
import tensorflow as tf
from sklearn import datasets
from analysis.gp import GPRNP
from analysis.gp_tf import GPR
from analysis.gp_tf import GPRGD
from analysis.gpr import gpr_models
from analysis.gpr.optimize import tf_optimize
from analysis.gpr.predict import gpflow_predict

# test numpy version GPR
class TestGPRNP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestGPRNP, cls).setUpClass()
        boston = datasets.load_boston()
        data = boston['data']
        X_train = data[0:500]
        X_test = data[500:]
        y_train = boston['target'][0:500].reshape(500, 1)
        cls.model = GPRNP(length_scale=1.0, magnitude=1.0)
        cls.model.fit(X_train, y_train, ridge=1.0)
        cls.gpr_result = cls.model.predict(X_test)

    def test_ypreds(self):
        ypreds_round = [round(x[0], 4) for x in self.gpr_result.ypreds]
        expected_ypreds = [0.0181, 0.0014, 0.0006, 0.0015, 0.0039, 0.0014]
        self.assertEqual(ypreds_round, expected_ypreds)

    def test_sigmas(self):
        sigmas_round = [round(x[0], 4) for x in self.gpr_result.sigmas]
        expected_sigmas = [1.4142, 1.4142, 1.4142, 1.4142, 1.4142, 1.4142]
        self.assertEqual(sigmas_round, expected_sigmas)


# test Tensorflow version GPR
class TestGPRTF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestGPRTF, cls).setUpClass()
        boston = datasets.load_boston()
        data = boston['data']
        X_train = data[0:500]
        X_test = data[500:]
        y_train = boston['target'][0:500].reshape(500, 1)
        cls.model = GPR(length_scale=1.0, magnitude=1.0, ridge=1.0)
        cls.model.fit(X_train, y_train)
        cls.gpr_result = cls.model.predict(X_test)

    def test_ypreds(self):
        ypreds_round = [round(x[0], 4) for x in self.gpr_result.ypreds]
        expected_ypreds = [0.0181, 0.0014, 0.0006, 0.0015, 0.0039, 0.0014]
        self.assertEqual(ypreds_round, expected_ypreds)

    def test_sigmas(self):
        sigmas_round = [round(x[0], 4) for x in self.gpr_result.sigmas]
        expected_sigmas = [1.4142, 1.4142, 1.4142, 1.4142, 1.4142, 1.4142]
        self.assertEqual(sigmas_round, expected_sigmas)


# test GPFlow version GPR
class TestGPRGPFlow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestGPRGPFlow, cls).setUpClass()
        boston = datasets.load_boston()
        data = boston['data']
        X_train = data[0:500]
        X_test = data[500:]
        y_train = boston['target'][0:500].reshape(500, 1)

        model_kwargs = {'lengthscales': 1, 'variance': 1, 'noise_variance': 1}
        tf.reset_default_graph()
        graph = tf.get_default_graph()
        gpflow.reset_default_session(graph=graph)
        cls.m = gpr_models.create_model('BasicGP', X=X_train, y=y_train,
                                        **model_kwargs)
        cls.gpr_result = gpflow_predict(cls.m.model, X_test)

    def test_ypreds(self):
        ypreds_round = [round(x[0], 4) for x in self.gpr_result.ypreds]
        expected_ypreds = [0.0181, 0.0014, 0.0006, 0.0015, 0.0039, 0.0014]
        self.assertEqual(ypreds_round, expected_ypreds)

    def test_sigmas(self):
        sigmas_round = [round(x[0], 4) for x in self.gpr_result.sigmas]
        expected_sigmas = [1.4142, 1.4142, 1.4142, 1.4142, 1.4142, 1.4142]
        self.assertEqual(sigmas_round, expected_sigmas)


# test GPFlow version Gradient Descent
class TestGDGPFlow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestGDGPFlow, cls).setUpClass()
        boston = datasets.load_boston()
        data = boston['data']
        X_train = data[0:500]
        X_test = data[500:501]
        y_train = boston['target'][0:500].reshape(500, 1)
        X_min = np.min(X_train, 0)
        X_max = np.max(X_train, 0)

        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        model_kwargs = {}
        opt_kwargs = {}
        opt_kwargs['learning_rate'] = 0.01
        opt_kwargs['maxiter'] = 10
        opt_kwargs['bounds'] = [X_min, X_max]
        opt_kwargs['ucb_beta'] = 1.0

        tf.reset_default_graph()
        graph = tf.get_default_graph()
        gpflow.reset_default_session(graph=graph)
        cls.m = gpr_models.create_model('BasicGP', X=X_train, y=y_train, **model_kwargs)
        cls.gpr_result = tf_optimize(cls.m.model, X_test, **opt_kwargs)

    def test_ypreds(self):
        ypreds_round = [round(x[0], 4) for x in self.gpr_result.ypreds]
        expected_ypreds = [0.5272]
        self.assertEqual(ypreds_round, expected_ypreds)

    def test_sigmas(self):
        sigmas_round = [round(x[0], 4) for x in self.gpr_result.sigmas]
        expected_sigmas = [1.4153]
        self.assertEqual(sigmas_round, expected_sigmas)


# test Tensorflow version Gradient Descent
class TestGDTF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestGDTF, cls).setUpClass()
        boston = datasets.load_boston()
        data = boston['data']
        X_train = data[0:500]
        X_test = data[500:501]
        y_train = boston['target'][0:500].reshape(500, 1)
        Xmin = np.min(X_train, 0)
        Xmax = np.max(X_train, 0)
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        cls.model = GPRGD(length_scale=2.0, magnitude=1.0, max_iter=10, learning_rate=0.01,
                          ridge=1.0, hyperparameter_trainable=True, sigma_multiplier=1.0)
        cls.model.fit(X_train, y_train, Xmin, Xmax)
        cls.gpr_result = cls.model.predict(X_test)

    def test_ypreds(self):
        ypreds_round = [round(x[0], 4) for x in self.gpr_result.ypreds]
        expected_ypreds = [0.5272]
        self.assertEqual(ypreds_round, expected_ypreds)

    def test_sigmas(self):
        sigmas_round = [round(x[0], 4) for x in self.gpr_result.sigmas]
        expected_sigmas = [1.4153]
        self.assertEqual(sigmas_round, expected_sigmas)

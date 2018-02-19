#
# OtterTune - test_gpr.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import unittest
from sklearn import datasets
from analysis.gp import GPRNP
from analysis.gp_tf import GPR


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

    def test_gprnp_ypreds(self):
        ypreds_round = [round(x[0], 4) for x in self.gpr_result.ypreds]
        expected_ypreds = [0.0181, 0.0014, 0.0006, 0.0015, 0.0039, 0.0014]
        self.assertEqual(ypreds_round, expected_ypreds)

    def test_gprnp_sigmas(self):
        sigmas_round = [round(x[0], 4) for x in self.gpr_result.sigmas]
        expected_sigmas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
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
        cls.model = GPR(length_scale=1.0, magnitude=1.0)
        cls.model.fit(X_train, y_train, ridge=1.0)
        cls.gpr_result = cls.model.predict(X_test)

    def test_gprnp_ypreds(self):
        ypreds_round = [round(x[0], 4) for x in self.gpr_result.ypreds]
        expected_ypreds = [0.0181, 0.0014, 0.0006, 0.0015, 0.0039, 0.0014]
        self.assertEqual(ypreds_round, expected_ypreds)

    def test_gprnp_sigmas(self):
        sigmas_round = [round(x[0], 4) for x in self.gpr_result.sigmas]
        expected_sigmas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.assertEqual(sigmas_round, expected_sigmas)

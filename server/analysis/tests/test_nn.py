#
# OtterTune - test_nn.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import random
import unittest
import numpy as np
from tensorflow import set_random_seed
from sklearn import datasets
from analysis.nn_tf import NeuralNet


# test neural network
class TestNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestNN, cls).setUpClass()
        boston = datasets.load_boston()
        data = boston['data']
        X_train = data[0:500]
        X_test = data[500:]
        y_train = boston['target'][0:500].reshape(500, 1)
        random.seed(0)
        np.random.seed(0)
        set_random_seed(0)
        cls.model = NeuralNet(n_input=X_test.shape[1],
                              batch_size=X_test.shape[0])
        cls.model.fit(X_train, y_train)
        cls.nn_result = cls.model.predict(X_test)
        cls.nn_recommend = cls.model.recommend(X_test)

    def test_nn_ypreds(self):
        ypreds_round = ['%.3f' % x[0] for x in self.nn_result]
        expected_ypreds = ['21.279', '22.668', '23.115', '27.228', '25.892', '23.967']
        self.assertEqual(ypreds_round, expected_ypreds)

    def test_nn_yrecommend(self):
        recommends_round = ['%.3f' % x[0] for x in self.nn_recommend.minl]
        expected_recommends = ['21.279', '21.279', '21.279', '21.279', '21.279', '21.279']
        self.assertEqual(recommends_round, expected_recommends)

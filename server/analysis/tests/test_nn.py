#
# OtterTune - test_nn.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import unittest
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
        set_random_seed(1)
        cls.model = NeuralNet(n_input=X_test.shape[1],
                              batch_size=X_test.shape[0])
        cls.model.fit(X_train, y_train)
        cls.nn_result = cls.model.predict(X_test)
        cls.nn_recommend = cls.model.recommend(X_test)

    def test_nn_ypreds(self):
        ypreds_round = ['%.3f' % x[0] for x in self.nn_result]
        expected_ypreds = ['20.503', '22.158', '22.158', '25.692', '24.536', '23.637']
        self.assertEqual(ypreds_round, expected_ypreds)

    def test_nn_yrecommend(self):
        recommends_round = ['%.3f' % x[0] for x in self.nn_recommend.minl]
        expected_recommends = ['14.229', '22.158', '22.158', '23.591', '23.591', '23.593']
        self.assertEqual(recommends_round, expected_recommends)

#
# OtterTune - test_ddpg.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

import random
import unittest
from sklearn import datasets
import numpy as np
import torch
from analysis.ddpg.ddpg import DDPG


# test ddpg model
class TestDDPG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        super(TestDDPG, cls).setUpClass()
        boston = datasets.load_boston()
        data = boston['data']
        X_train = data[0:500]
        cls.X_test = data[500:]
        y_train = boston['target'][0:500].reshape(500, 1)
        cls.ddpg = DDPG(n_actions=1, n_states=13)
        for i in range(500):
            knob_data = np.array([random.random()])
            prev_metric_data = X_train[i - 1]
            metric_data = X_train[i]
            reward = y_train[i - 1]
            cls.ddpg.add_sample(prev_metric_data, knob_data, reward, metric_data, False)
            if len(cls.ddpg.replay_memory) > 32:
                cls.ddpg.update()

    def test_ddpg_ypreds(self):
        ypreds_round = [round(self.ddpg.choose_action(x)[0], 4) for x in self.X_test]
        expected_ypreds = [0.1778, 0.1914, 0.2607, 0.4459, 0.5660, 0.3836]
        self.assertEqual(ypreds_round, expected_ypreds)
        for ypred_round, expected_ypred in zip(ypreds_round, expected_ypreds):
            self.assertAlmostEqual(ypred_round, expected_ypred, places=6)

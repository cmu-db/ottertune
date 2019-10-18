#
# OtterTune - test_ddpg.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

import random
import unittest
import numpy as np
import torch
from analysis.ddpg.ddpg import DDPG


# test ddpg model:
# The enviroment has 1-dim state and 1-dim action, the reward is calculated as follows:
# if state < 0.5, taking action < 0.5 gets reward 1, taking action >= 0.5 gets reward 0
# if state >= 0.5, taking action >= 0.5 gets reward 1, taking action < 0.5 gets reward 0
# Train 500 iterations and test for 500 iterations
# If the average reward during test is larger than 0.9, this test passes
class TestDDPG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        super(TestDDPG, cls).setUpClass()
        cls.ddpg = DDPG(n_actions=1, n_states=1, gamma=0)
        for _ in range(700):
            knob_data = np.array([random.random()])
            prev_metric_data = np.array([random.random()])
            metric_data = np.array([random.random()])
            reward = 1.0 if (prev_metric_data[0] - 0.5) * (knob_data[0] - 0.5) > 0 else 0.0
            reward = np.array([reward])
            cls.ddpg.add_sample(prev_metric_data, knob_data, reward, metric_data, False)
            if len(cls.ddpg.replay_memory) > 32:
                cls.ddpg.update()

    def test_ddpg_ypreds(self):
        total_reward = 0.0
        for _ in range(500):
            prev_metric_data = np.array([random.random()])
            knob_data = self.ddpg.choose_action(prev_metric_data)
            reward = 1.0 if (prev_metric_data[0] - 0.5) * (knob_data[0] - 0.5) > 0 else 0.0
            total_reward += reward
        self.assertGreater(total_reward / 500, 0.9)


if __name__ == '__main__':
    unittest.main()

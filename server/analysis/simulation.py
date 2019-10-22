#
# OtterTune - simulation.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#


import imp
import random
import os
import sys
try:
    imp.find_module('matplotlib.pyplot')
    import matplotlib.pyplot as plt
except (ModuleNotFoundError, ImportError):
    plt = None
import numpy as np
import torch
sys.path.append("../")
from analysis.util import get_analysis_logger  # noqa
from analysis.ddpg.ddpg import DDPG  # noqa

LOG = get_analysis_logger(__name__)


class Environment(object):
    def __init__(self, n_knob, n_metric, mode=0):
        self.knob_dim = n_knob
        self.metric_dim = n_metric
        self.mode = mode

    def identity_sqrt(self, knob_data):
        n1 = self.knob_dim // 4
        n2 = self.knob_dim // 4
        part1 = np.sum(knob_data[0: n1])
        part2 = np.sum(np.sqrt(knob_data[n1: n1 + n2]))
        reward = np.array([part1 + part2]) / (self.knob_dim // 2)
        metric_data = np.zeros(self.metric_dim)
        return reward, metric_data

    def borehole(self, knob_data):
        # ref: http://www.sfu.ca/~ssurjano/borehole.html
        # pylint: disable=invalid-name
        rw = knob_data[0] * (0.15 - 0.05) + 0.05
        r = knob_data[1] * (50000 - 100) + 100
        Tu = knob_data[2] * (115600 - 63070) + 63070
        Hu = knob_data[3] * (1110 - 990) + 990
        Tl = knob_data[4] * (116 - 63.1) + 63.1
        Hl = knob_data[5] * (820 - 700) + 700
        L = knob_data[6] * (1680 - 1120) + 1120
        Kw = knob_data[7] * (12045 - 9855) + 9855

        frac = 2 * L * Tu / (np.log(r / rw) * rw ** 2 * Kw)
        reward = 2 * np.pi * Tu * (Hu - Hl) / (np.log(r / rw) * (1 + frac + Tu / Tl))
        return np.array([reward]), np.zeros(self.metric_dim)

    def threshold(self, knob_data):
        n1 = self.knob_dim // 4
        n2 = self.knob_dim // 4
        part1 = np.sum(knob_data[0: n1] > 0.9)
        part2 = np.sum(knob_data[n1: n1 + n2] < 0.1)
        reward = np.array([part1 + part2])
        metric_data = np.zeros(self.metric_dim)
        return reward, metric_data

    def simulate(self, knob_data):
        if self.mode == 0:
            return self.identity_sqrt(knob_data)
        elif self.mode == 1:
            return self.threshold(knob_data)
        elif self.mode == 2:
            return self.borehole(knob_data)


def train_ddpg(env, gamma=0.99, tau=0.002, lr=0.01, batch_size=32, n_loops=1000):
    results = []
    x_axis = []
    ddpg = DDPG(n_actions=env.knob_dim, n_states=env.metric_dim, gamma=gamma, tau=tau,
                clr=lr, alr=lr, batch_size=batch_size)
    knob_data = np.random.rand(env.knob_dim)
    prev_metric_data = np.zeros(env.metric_dim)
    for i in range(n_loops):
        reward, metric_data = env.simulate(knob_data)
        ddpg.add_sample(prev_metric_data, knob_data, reward, metric_data, False)
        ddpg.update()
        if i % 20 == 0:
            results.append(run_ddpg(env, ddpg))
            x_axis.append(i)
        prev_metric_data = metric_data
        knob_data = ddpg.choose_action(prev_metric_data)
    return np.array(results), np.array(x_axis)


def run_ddpg(env, ddpg):
    total_reward = 0.0
    n_samples = 100
    prev_metric_data = np.zeros(env.metric_dim)
    for _ in range(n_samples):
        knob_data = ddpg.choose_action(prev_metric_data)
        reward, prev_metric_data = env.simulate(knob_data)
        total_reward += reward
    return total_reward / n_samples


def plotlines(x_axis, data1, data2, label1, label2, title, path):
    if plt:
        plt.plot(x_axis, data1, color='red', label=label1)
        plt.plot(x_axis, data2, color='blue', label=label2)
        plt.legend()
        plt.xlabel("loops")
        plt.ylabel("rewards")
        plt.title(title)
        plt.savefig(path)
        plt.clf()


def main(knob_dim=8, metric_dim=60, lr=0.0001, mode=2, n_loops=2000):
    if not plt:
        LOG.info("Cannot import matplotlib. Will write results to files instead of figures.")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = Environment(knob_dim, metric_dim, mode=mode)

    n_repeats = 10
    for i in range(n_repeats):
        if i == 0:
            results1, x_axis = train_ddpg(env, gamma=0, lr=lr, n_loops=n_loops)
        else:
            results1 += train_ddpg(env, gamma=0, lr=lr, n_loops=n_loops)[0]
    for i in range(n_repeats):
        if i == 0:
            results2, x_axis = train_ddpg(env, gamma=0.99, lr=lr, n_loops=n_loops)
        else:
            results2 += train_ddpg(env, gamma=0.99, lr=lr, n_loops=n_loops)[0]
    results1 /= n_repeats
    results2 /= n_repeats
    title = "knob_{}_lr_{}".format(knob_dim, lr)
    if plt:
        if not os.path.exists("figures"):
            os.mkdir("figures")
        filename = "figures/{}.pdf".format(title)
        plotlines(x_axis, results1, results2, "gamma=0", "gamma=0.99", title, filename)
    else:
        with open(title + '_1.csv', 'w') as f1:
            for i, result in zip(x_axis, results1):
                f1.write(str(i) + ',' + str(result[0]) + '\n')
        with open(title + '_2.csv', 'w') as f2:
            for i, result in zip(x_axis, results2):
                f2.write(str(i) + ',' + str(result[0]) + '\n')


if __name__ == '__main__':
    main()

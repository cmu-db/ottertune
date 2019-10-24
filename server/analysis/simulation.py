#
# OtterTune - simulation.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#


import heapq
import random
import os
import sys
try:
    import matplotlib.pyplot as plt
except (ModuleNotFoundError, ImportError):
    plt = None
import numpy as np
import torch
sys.path.append("../")
from analysis.util import get_analysis_logger  # noqa
from analysis.ddpg.ddpg import DDPG  # noqa
from analysis.gp_tf import GPRGD  # noqa
from analysis.nn_tf import NeuralNet  # noqa

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


def ddpg(env, config, n_loops=1000):
    results = []
    x_axis = []
    gamma = config['gamma']
    tau = config['tau']
    lr = config['lr']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    model_ddpg = DDPG(n_actions=env.knob_dim, n_states=env.metric_dim, gamma=gamma, tau=tau,
                      clr=lr, alr=lr, batch_size=batch_size)
    knob_data = np.random.rand(env.knob_dim)
    prev_metric_data = np.zeros(env.metric_dim)
    for i in range(n_loops):
        reward, metric_data = env.simulate(knob_data)
        model_ddpg.add_sample(prev_metric_data, knob_data, reward, metric_data)
        for _ in range(n_epochs):
            model_ddpg.update()
        results.append(reward)
        x_axis.append(i)
        prev_metric_data = metric_data
        knob_data = model_ddpg.choose_action(prev_metric_data)
    return np.array(results), np.array(x_axis)


class ReplayMemory(object):

    def __init__(self):
        self.actions = []
        self.rewards = []

    def push(self, action, reward):
        self.actions.append(action.tolist())
        self.rewards.append(reward.tolist())

    def get_all(self):
        return self.actions, self.rewards


def dnn(env, config, n_loops=100):
    results = []
    x_axis = []
    memory = ReplayMemory()
    num_samples = config['num_samples']
    Xmin = np.zeros(env.knob_dim)
    Xmax = np.ones(env.knob_dim)
    for i in range(n_loops):
        X_samples = np.random.rand(num_samples, env.knob_dim)
        if i >= 10:
            actions, rewards = memory.get_all()
            tuples = tuple(zip(actions, rewards))
            top10 = heapq.nlargest(10, tuples, key=lambda e: e[1])
            for entry in top10:
                X_samples = np.vstack((X_samples, np.array(entry[0])))
        model_nn = NeuralNet(n_input=X_samples.shape[1],
                             batch_size=X_samples.shape[0],
                             explore_iters=500,
                             noise_scale_begin=0.1,
                             noise_scale_end=0.0,
                             debug=False,
                             debug_interval=100)
        if i >= 5:
            actions, rewards = memory.get_all()
            model_nn.fit(np.array(actions), -np.array(rewards), fit_epochs=500)
        res = model_nn.recommend(X_samples, Xmin, Xmax,
                                 explore=500, recommend_epochs=500)
        best_config_idx = np.argmin(res.minl.ravel())
        best_config = res.minl_conf[best_config_idx, :]
        reward, _ = env.simulate(best_config)
        memory.push(best_config, reward)
        LOG.info('loop: %d reward: %f', i, reward[0])
        results.append(reward)
        x_axis.append(i)
    return np.array(results), np.array(x_axis)


def gprgd(env, config, n_loops=100):
    results = []
    x_axis = []
    memory = ReplayMemory()
    num_samples = config['num_samples']
    X_min = np.zeros(env.knob_dim)
    X_max = np.ones(env.knob_dim)
    for _ in range(5):
        action = np.random.rand(env.knob_dim)
        reward, _ = env.simulate(action)
        memory.push(action, reward)
    for i in range(n_loops):
        X_samples = np.random.rand(num_samples, env.knob_dim)
        if i >= 5:
            actions, rewards = memory.get_all()
            tuples = tuple(zip(actions, rewards))
            top10 = heapq.nlargest(10, tuples, key=lambda e: e[1])
            for entry in top10:
                # Tensorflow get broken if we use the training data points as
                # starting points for GPRGD.
                X_samples = np.vstack((X_samples, np.array(entry[0]) * 0.97 + 0.01))
        model = GPRGD(length_scale=1.0,
                      magnitude=1.0,
                      max_train_size=7000,
                      batch_size=3000,
                      num_threads=4,
                      learning_rate=0.01,
                      epsilon=1e-6,
                      max_iter=500,
                      sigma_multiplier=3.0,
                      mu_multiplier=1.0)

        actions, rewards = memory.get_all()
        model.fit(np.array(actions), -np.array(rewards), X_min, X_max, ridge=0.01)
        res = model.predict(X_samples)
        best_config_idx = np.argmin(res.minl.ravel())
        best_config = res.minl_conf[best_config_idx, :]
        reward, _ = env.simulate(best_config)
        memory.push(best_config, reward)
        LOG.info('loop: %d reward: %f', i, reward[0])
        results.append(reward)
        x_axis.append(i)
    return np.array(results), np.array(x_axis)


def plotlines(x_axis, results, labels, title, path):
    if plt:
        for result, label in zip(results, labels):
            plt.plot(x_axis, result, label=label)
        plt.legend()
        plt.xlabel("loops")
        plt.ylabel("rewards")
        plt.title(title)
        plt.savefig(path)
        plt.clf()


def run(tuners, configs, labels, knob_dim, metric_dim, mode, n_loops, n_repeats):
    if not plt:
        LOG.info("Cannot import matplotlib. Will write results to files instead of figures.")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = Environment(knob_dim, metric_dim, mode=mode)
    results = []
    for i in range(n_repeats):
        for j, _ in enumerate(tuners):
            result, x_axis = tuners[j](env, configs[j], n_loops=n_loops)
            if i is 0:
                results.append(result / n_repeats)
            else:
                results[j] += result / n_repeats

    title = "mode_{}_knob_{}".format(mode, knob_dim)

    if plt:
        if not os.path.exists("figures"):
            os.mkdir("figures")
        filename = "figures/{}.pdf".format(title)
        plotlines(x_axis, results, labels, title, filename)
    for j in range(len(tuners)):
        with open(title + '_' + labels[j] + '.csv', 'w') as f:
            for i, result in zip(x_axis, results[j]):
                f.write(str(i) + ',' + str(result[0]) + '\n')


def main():
    knob_dim = 192
    metric_dim = 60
    mode = 0
    n_loops = 2
    n_repeats = 1
    configs = [{'gamma': 0., 'tau': 0.002, 'lr': 0.001, 'batch_size': 32, 'n_epochs': 30},
               {'gamma': 0.99, 'tau': 0.002, 'lr': 0.001, 'batch_size': 32, 'n_epochs': 30},
               {'num_samples': 30},
               {'num_samples': 30}]
    tuners = [ddpg, ddpg, dnn, gprgd]
    labels = [tuner.__name__ for tuner in tuners]
    labels[0] += '_gamma_0'
    labels[1] += '_gamma_99'
    run(tuners, configs, labels, knob_dim, metric_dim, mode, n_loops, n_repeats)


if __name__ == '__main__':
    main()

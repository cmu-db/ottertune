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
from analysis.util import get_analysis_logger, TimerStruct  # noqa
from analysis.ddpg.ddpg import DDPG  # noqa
from analysis.ddpg.ou_process import OUProcess  # noqa
from analysis.gp_tf import GPRGD  # noqa
from analysis.nn_tf import NeuralNet  # noqa
from analysis.gpr import gpr_models  # noqa
from analysis.gpr import ucb  # noqa
from analysis.gpr.optimize import tf_optimize  # noqa

LOG = get_analysis_logger(__name__)


class Environment(object):
    def __init__(self, knob_dim, metric_dim, modes=[0], reward_variance=0,
                 metrics_variance=0.2):
        self.knob_dim = knob_dim
        self.metric_dim = metric_dim
        self.modes = modes
        self.mode = np.random.choice(self.modes)
        self.counter = 0
        self.reward_variance = reward_variance
        self.metrics_variance = metrics_variance

    def identity_sqrt(self, knob_data):
        n1 = self.knob_dim // 4
        n2 = self.knob_dim // 4
        part1 = np.sum(knob_data[0: n1])
        part2 = np.sum(np.sqrt(knob_data[n1: n1 + n2]))
        reward = np.array([part1 + part2]) / (self.knob_dim // 2)
        return reward

    def threshold(self, knob_data):
        n1 = self.knob_dim // 4
        n2 = self.knob_dim // 4
        part1 = np.sum(knob_data[0: n1] > 0.9)
        part2 = np.sum(knob_data[n1: n1 + n2] < 0.1)
        reward = np.array([part1 + part2]) / (self.knob_dim // 2)
        return reward

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
        reward = 2 * np.pi * Tu * (Hu - Hl) / (np.log(r / rw) * (1 + frac + Tu / Tl)) / 310
        return np.array([reward])

    def get_metrics(self, mode):
        metrics = np.ones(self.metric_dim) * mode
        metrics += np.random.rand(self.metric_dim) * self.metrics_variance
        return metrics

    def simulate_mode(self, knob_data, mode):
        if mode == 0:
            reward = self.identity_sqrt(knob_data)
        elif mode == 1:
            reward = self.threshold(knob_data)
        elif mode == 2:
            reward = np.zeros(1)
            for i in range(0, len(knob_data), 8):
                reward += self.borehole(knob_data[i: i+8])[0] / len(knob_data) * 8
        reward = reward * (1.0 + self.reward_variance * np.random.rand(1)[0])
        return reward, self.get_metrics(mode)

    def simulate(self, knob_data):
        self.counter += 1
        k = 1
        # every k runs, sample a new workload
        if self.counter >= k:
            self.counter = 0
            self.mode = np.random.choice(self.modes)
        return self.simulate_mode(knob_data, self.mode)


def ddpg(env, config, n_loops=100):
    results = []
    x_axis = []
    num_collections = config['num_collections']
    gamma = config['gamma']
    a_lr = config['a_lr']
    c_lr = config['c_lr']
    n_epochs = config['n_epochs']
    model_ddpg = DDPG(n_actions=env.knob_dim, n_states=env.metric_dim, gamma=gamma,
                      clr=c_lr, alr=a_lr, shift=0.1)
    knob_data = np.random.rand(env.knob_dim)
    prev_metric_data = np.zeros(env.metric_dim)

    for i in range(num_collections):
        action = np.random.rand(env.knob_dim)
        reward, metric_data = env.simulate(action)
        if i > 0:
            model_ddpg.add_sample(prev_metric_data, prev_knob_data, prev_reward, metric_data)
        prev_metric_data = metric_data
        prev_knob_data = knob_data
        prev_reward = reward

    for i in range(n_loops):
        reward, metric_data = env.simulate(knob_data)
        model_ddpg.add_sample(prev_metric_data, prev_knob_data, prev_reward, metric_data)
        prev_metric_data = metric_data
        prev_knob_data = knob_data
        prev_reward = reward
        for _ in range(n_epochs):
            model_ddpg.update()
        results.append(reward)
        x_axis.append(i+1)
        LOG.info('loop: %d reward: %f', i, reward[0])
        knob_data = model_ddpg.choose_action(metric_data)
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
    num_collections = config['num_collections']
    num_samples = config['num_samples']
    ou_process = False
    Xmin = np.zeros(env.knob_dim)
    Xmax = np.ones(env.knob_dim)
    noise = OUProcess(env.knob_dim)

    for _ in range(num_collections):
        action = np.random.rand(env.knob_dim)
        reward, _ = env.simulate(action)
        memory.push(action, reward)

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
                             learning_rate=0.01,
                             explore_iters=100,
                             noise_scale_begin=0.1,
                             noise_scale_end=0.0,
                             debug=False,
                             debug_interval=100)
        actions, rewards = memory.get_all()
        model_nn.fit(np.array(actions), -np.array(rewards), fit_epochs=50)
        res = model_nn.recommend(X_samples, Xmin, Xmax, recommend_epochs=10, explore=False)
        best_config_idx = np.argmin(res.minl.ravel())
        best_config = res.minl_conf[best_config_idx, :]
        if ou_process:
            best_config += noise.noise()
            best_config = best_config.clip(0, 1)
        reward, _ = env.simulate(best_config)
        memory.push(best_config, reward)
        LOG.info('loop: %d reward: %f', i, reward[0])
        results.append(reward)
        x_axis.append(i+1)
    return np.array(results), np.array(x_axis)


def gpr(env, config, n_loops=100):
    results = []
    x_axis = []
    memory = ReplayMemory()
    num_collections = config['num_collections']
    num_samples = config['num_samples']
    X_min = np.zeros(env.knob_dim)
    X_max = np.ones(env.knob_dim)
    for _ in range(num_collections):
        action = np.random.rand(env.knob_dim)
        reward, _ = env.simulate(action)
        memory.push(action, reward)

    for i in range(n_loops):
        X_samples = np.random.rand(num_samples, env.knob_dim)
        if i >= 10:
            actions, rewards = memory.get_all()
            tuples = tuple(zip(actions, rewards))
            top10 = heapq.nlargest(10, tuples, key=lambda e: e[1])
            for entry in top10:
                # Tensorflow get broken if we use the training data points as
                # starting points for GPRGD.
                X_samples = np.vstack((X_samples, np.array(entry[0]) * 0.97 + 0.01))
        model = GPRGD(length_scale=1.0,
                      magnitude=1.0,
                      max_train_size=2000,
                      batch_size=100,
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
        x_axis.append(i+1)
    return np.array(results), np.array(x_axis)


def run_optimize(X, y, X_sample, model_name, opt_kwargs, model_kwargs):
    timer = TimerStruct()

    # Create model (this also optimizes the hyperparameters if that option is enabled
    timer.start()
    m = gpr_models.create_model(model_name, X=X, y=y, **model_kwargs)
    timer.stop()
    model_creation_sec = timer.elapsed_seconds
    LOG.info(m._model.as_pandas_table())

    # Optimize the DBMS's configuration knobs
    timer.start()
    X_new, ypred, yvar, loss = tf_optimize(m._model, X_sample, **opt_kwargs)
    timer.stop()
    config_optimize_sec = timer.elapsed_seconds

    return X_new, ypred, m.get_model_parameters(), m.get_hyperparameters()


def gpr_new(env, config, n_loops=100):
    model_name = 'BasicGP'
    model_opt_frequency = 5
    model_kwargs = {}
    model_kwargs['model_learning_rate'] = 0.001
    model_kwargs['model_maxiter'] = 5000
    opt_kwargs = {}
    opt_kwargs['learning_rate'] = 0.001
    opt_kwargs['maxiter'] = 100
    opt_kwargs['ucb_beta'] = 3.0

    results = []
    x_axis = []
    memory = ReplayMemory()
    num_samples = config['num_samples']
    num_collections = config['num_collections']
    X_min = np.zeros(env.knob_dim)
    X_max = np.ones(env.knob_dim)
    X_bounds = [X_min, X_max]
    opt_kwargs['bounds'] = X_bounds

    for _ in range(num_collections):
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

        actions, rewards = memory.get_all()

        ucb_beta = opt_kwargs.pop('ucb_beta')
        opt_kwargs['ucb_beta'] = ucb.get_ucb_beta(ucb_beta, t=i + 1., ndim=env.knob_dim)
        if model_opt_frequency > 0:
            optimize_hyperparams = i % model_opt_frequency == 0
            if not optimize_hyperparams:
                model_kwargs['hyperparameters'] = hyperparameters
        else:
            optimize_hyperparams = False
            model_kwargs['hyperparameters'] = None
        model_kwargs['optimize_hyperparameters'] = optimize_hyperparams

        X_new, ypred, model_params, hyperparameters = run_optimize(np.array(actions),
                                                                   -np.array(rewards),
                                                                   X_samples,
                                                                   model_name,
                                                                   opt_kwargs,
                                                                   model_kwargs)

        sort_index = np.argsort(ypred.squeeze())
        X_new = X_new[sort_index]
        ypred = ypred[sort_index].squeeze()

        action = X_new[0]
        reward, _ = env.simulate(action)
        memory.push(action, reward)
        LOG.info('loop: %d reward: %f', i, reward[0])
        results.append(reward)
        x_axis.append(i+1)

    return np.array(results), np.array(x_axis)


def plotlines(xs, results, labels, title, path):
    if plt:
        figsize = 13, 10
        figure, ax = plt.subplots(figsize=figsize)
        lines = []
        N = 1
        weights = np.ones(N)
        for x_axis, result, label in zip(xs, results, labels):
            result = np.convolve(weights/weights.sum(), result.flatten())[N-1:-N+1]
            lines.append(plt.plot(x_axis[:-N+1], result, label=label, lw=4)[0])
        plt.legend(handles=lines, fontsize=30)
        plt.title(title, fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax.set_xlabel("loops", fontsize=30)
        ax.set_ylabel("rewards", fontsize=30)
        plt.savefig(path)
        plt.clf()


def run(tuners, configs, labels, title, env, n_loops, n_repeats):
    if not plt:
        LOG.info("Cannot import matplotlib. Will write results to files instead of figures.")
    random.seed(0)
    np.random.seed(1)
    torch.manual_seed(0)
    results = []
    xs = []
    for j, _ in enumerate(tuners):
        for i in range(n_repeats[j]):
            result, x_axis = tuners[j](env, configs[j], n_loops=n_loops)
            if i is 0:
                results.append(result / n_repeats[j])
                xs.append(x_axis)
            else:
                results[j] += result / n_repeats[j]

    if plt:
        if not os.path.exists("simulation_figures"):
            os.mkdir("simulation_figures")
        filename = "simulation_figures/{}.pdf".format(title)
        plotlines(xs, results, labels, title, filename)
    if not os.path.exists("simulation_results"):
        os.mkdir("simulation_results")
    for j in range(len(tuners)):
        with open("simulation_results/" + title + '_' + labels[j] + '.csv', 'w') as f:
            for i, result in zip(xs[j], results[j]):
                f.write(str(i) + ',' + str(result[0]) + '\n')


def main():
    env = Environment(knob_dim=24, metric_dim=60, modes=[2], reward_variance=0.05)
    title = 'compare'
    n_repeats = [1, 1, 1, 1]
    n_loops = 80
    configs = [{'gamma': 0., 'c_lr': 0.001, 'a_lr': 0.01, 'num_collections': 50, 'n_epochs': 50},
               {'num_samples': 30, 'num_collections': 50},
               {'num_samples': 30, 'num_collections': 50},
               {'num_samples': 30, 'num_collections': 50}]
    tuners = [ddpg, gpr_new, dnn, gpr]
    labels = [tuner.__name__ for tuner in tuners]
    run(tuners, configs, labels, title, env, n_loops, n_repeats)


if __name__ == '__main__':
    main()

#
# OtterTune - nn_tf.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Sep 16, 2019
@author: Bohan Zhang
'''

import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .util import get_analysis_logger

LOG = get_analysis_logger(__name__)


class NeuralNetResult(object):
    def __init__(self, minl=None, minl_conf=None):
        self.minl = minl
        self.minl_conf = minl_conf


class NeuralNet(object):

    def __init__(self,
                 n_input,
                 learning_rate=0.01,
                 debug=False,
                 debug_interval=100,
                 batch_size=1,
                 explore_iters=500,
                 noise_scale_begin=0.1,
                 noise_scale_end=0):

        self.history = None
        self.recommend_iters = 0
        self.n_input = n_input
        self.debug = debug
        self.debug_interval = debug_interval
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.explore_iters = explore_iters
        self.noise_scale_begin = noise_scale_begin
        self.noise_scale_end = noise_scale_end
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # input X is placeholder, weights are variables.
        self.model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.relu, input_shape=[n_input]),
            layers.Dropout(0.5),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(1)
        ])
        self.model.compile(loss='mean_squared_error',
                           optimizer=self.optimizer,
                           metrics=['mean_squared_error', 'mean_absolute_error'])
        self.vars = {}
        self.ops = {}
        self.build_graph()

    def save_weights(self, weights_file):
        self.model.save_weights(weights_file)

    def load_weights(self, weights_file):
        try:
            self.model.load_weights(weights_file)
            if self.debug:
                LOG.info('Neural Network Model weights file exists, load weights from the file')
        except Exception:  # pylint: disable=broad-except
            LOG.info('Weights file does not match neural network model, train model from scratch')

    def get_weights_bin(self):
        return pickle.dumps(self.model.get_weights())

    def set_weights_bin(self, weights):
        try:
            self.model.set_weights(pickle.loads(weights))
            if self.debug:
                LOG.info('Neural Network Model weights exists, load the existing weights')
        except Exception:  # pylint: disable=broad-except
            LOG.info('Weights does not match neural network model, train model from scratch')

    # Build same neural network as self.model, But input X is variables,
    # weights are placedholders. Find optimial X using gradient descent.
    def build_graph(self):
        batch_size = self.batch_size
        self.graph = tf.Graph()
        with self.graph.as_default():
            x_ = tf.Variable(tf.ones([batch_size, self.n_input]))
            w1_ = tf.placeholder(tf.float32, [self.n_input, 64])
            b1_ = tf.placeholder(tf.float32, [64])
            w2_ = tf.placeholder(tf.float32, [64, 64])
            b2_ = tf.placeholder(tf.float32, [64])
            w3_ = tf.placeholder(tf.float32, [64, 1])
            b3_ = tf.placeholder(tf.float32, [1])
            l1_ = tf.nn.relu(tf.add(tf.matmul(x_, w1_), b1_))
            l2_ = tf.nn.relu(tf.add(tf.matmul(l1_, w2_), b2_))
            y_ = tf.add(tf.matmul(l2_, w3_), b3_)
            optimizer_ = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_ = optimizer_.minimize(y_)

            self.vars['x_'] = x_
            self.vars['y_'] = y_
            self.vars['w1_'] = w1_
            self.vars['w2_'] = w2_
            self.vars['w3_'] = w3_
            self.vars['b1_'] = b1_
            self.vars['b2_'] = b2_
            self.vars['b3_'] = b3_
            self.ops['train_'] = train_

    def fit(self, X_train, y_train, fit_epochs=500):
        self.history = self.model.fit(
            X_train, y_train, epochs=fit_epochs, verbose=0)
        if self.debug:
            mse = self.history.history['mean_squared_error']
            i = 0
            size = len(mse)
            while(i < size):
                LOG.info("Neural network training phase, epoch %d: mean_squared_error %f",
                         i, mse[i])
                i += self.debug_interval
            LOG.info("Neural network training phase, epoch %d: mean_squared_error %f",
                     size - 1, mse[size - 1])

    def predict(self, X_pred):
        return self.model.predict(X_pred)

    # Reference: Parameter Space Noise for Exploration.ICLR 2018, https://arxiv.org/abs/1706.01905
    def add_noise(self, weights):
        scale = self.adaptive_noise_scale()
        size = weights.shape[-1]
        noise = scale * np.random.normal(size=size)
        return weights + noise

    def adaptive_noise_scale(self):
        if self.recommend_iters > self.explore_iters:
            scale = self.noise_scale_end
        else:
            scale = self.noise_scale_begin - (self.noise_scale_begin - self.noise_scale_end) \
                * 1.0 * self.recommend_iters / self.explore_iters
        return scale

    def recommend(self, X_start, X_min=None, X_max=None, recommend_epochs=500, explore=False):
        batch_size = len(X_start)
        assert(batch_size == self.batch_size)
        w1, b1 = self.model.get_layer(index=0).get_weights()
        w2, b2 = self.model.get_layer(index=2).get_weights()
        w3, b3 = self.model.get_layer(index=3).get_weights()

        if explore is True:
            w1 = self.add_noise(w1)
            b1 = self.add_noise(b1)
            w2 = self.add_noise(w2)
            b2 = self.add_noise(b2)
            w3 = self.add_noise(w3)
            b3 = self.add_noise(b3)

        y_predict = self.predict(X_start)
        if self.debug:
            LOG.info("Recommend phase, y prediction: min %f, max %f, mean %f",
                     np.min(y_predict), np.max(y_predict), np.mean(y_predict))

        with tf.Session(graph=self.graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            assign_x_op = self.vars['x_'].assign(X_start)
            sess.run(assign_x_op)
            y_before = sess.run(self.vars['y_'],
                                feed_dict={self.vars['w1_']: w1, self.vars['w2_']: w2,
                                           self.vars['w3_']: w3, self.vars['b1_']: b1,
                                           self.vars['b2_']: b2, self.vars['b3_']: b3})
            if self.debug:
                LOG.info("Recommend phase, y before gradient descent: min %f, max %f, mean %f",
                         np.min(y_before), np.max(y_before), np.mean(y_before))

            for i in range(recommend_epochs):
                sess.run(self.ops['train_'],
                         feed_dict={self.vars['w1_']: w1, self.vars['w2_']: w2,
                                    self.vars['w3_']: w3, self.vars['b1_']: b1,
                                    self.vars['b2_']: b2, self.vars['b3_']: b3})

                # constrain by X_min and X_max
                if X_min is not None and X_max is not None:
                    X_train = sess.run(self.vars['x_'])
                    X_train = np.minimum(X_train, X_max)
                    X_train = np.maximum(X_train, X_min)
                    constraint_x_op = self.vars['x_'].assign(X_train)
                    sess.run(constraint_x_op)

                if self.debug and i % self.debug_interval == 0:
                    y_train = sess.run(self.vars['y_'],
                                       feed_dict={self.vars['w1_']: w1, self.vars['w2_']: w2,
                                                  self.vars['w3_']: w3, self.vars['b1_']: b1,
                                                  self.vars['b2_']: b2, self.vars['b3_']: b3})
                    LOG.info("Recommend phase, epoch %d, y: min %f, max %f, mean %f",
                             i, np.min(y_train), np.max(y_train), np.mean(y_train))

            y_recommend = sess.run(self.vars['y_'],
                                   feed_dict={self.vars['w1_']: w1, self.vars['w2_']: w2,
                                              self.vars['w3_']: w3, self.vars['b1_']: b1,
                                              self.vars['b2_']: b2, self.vars['b3_']: b3})
            X_recommend = sess.run(self.vars['x_'])
            res = NeuralNetResult(minl=y_recommend, minl_conf=X_recommend)

            if self.debug:
                LOG.info("Recommend phase, epoch %d, y after gradient descent: \
                         min %f, max %f, mean %f", recommend_epochs, np.min(y_recommend),
                         np.max(y_recommend), np.mean(y_recommend))

        self.recommend_iters += 1
        return res

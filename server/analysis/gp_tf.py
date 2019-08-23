#
# OtterTune - gp_tf.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Aug 18, 2016

@author: Bohan Zhang, Dana Van Aken
'''

import gc
import numpy as np
import tensorflow as tf

from .util import get_analysis_logger

LOG = get_analysis_logger(__name__)


class GPRResult(object):

    def __init__(self, ypreds=None, sigmas=None):
        self.ypreds = ypreds
        self.sigmas = sigmas


class GPRGDResult(GPRResult):

    def __init__(self, ypreds=None, sigmas=None,
                 minl=None, minl_conf=None):
        super(GPRGDResult, self).__init__(ypreds, sigmas)
        self.minl = minl
        self.minl_conf = minl_conf


class GPR(object):

    def __init__(self, length_scale=1.0, magnitude=1.0, max_train_size=7000,
                 batch_size=3000, num_threads=4, check_numerics=True, debug=False):
        assert np.isscalar(length_scale)
        assert np.isscalar(magnitude)
        assert length_scale > 0 and magnitude > 0
        self.length_scale = length_scale
        self.magnitude = magnitude
        self.max_train_size_ = max_train_size
        self.batch_size_ = batch_size
        self.num_threads_ = num_threads
        self.check_numerics = check_numerics
        self.debug = debug
        self.X_train = None
        self.y_train = None
        self.xy_ = None
        self.K = None
        self.K_inv = None
        self.graph = None
        self.vars = None
        self.ops = None

    def build_graph(self):
        self.vars = {}
        self.ops = {}
        self.graph = tf.Graph()
        with self.graph.as_default():
            mag_const = tf.constant(self.magnitude,
                                    dtype=np.float32,
                                    name='magnitude')
            ls_const = tf.constant(self.length_scale,
                                   dtype=np.float32,
                                   name='length_scale')

            # Nodes for distance computation
            v1 = tf.placeholder(tf.float32, name="v1")
            v2 = tf.placeholder(tf.float32, name="v2")
            dist_op = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2), 1), name='dist_op')
            if self.check_numerics:
                dist_op = tf.check_numerics(dist_op, "dist_op: ")

            self.vars['v1_h'] = v1
            self.vars['v2_h'] = v2
            self.ops['dist_op'] = dist_op

            # Nodes for kernel computation
            X_dists = tf.placeholder(tf.float32, name='X_dists')
            ridge_ph = tf.placeholder(tf.float32, name='ridge')
            K_op = mag_const * tf.exp(-X_dists / ls_const)
            if self.check_numerics:
                K_op = tf.check_numerics(K_op, "K_op: ")
            K_ridge_op = K_op + tf.diag(ridge_ph)
            if self.check_numerics:
                K_ridge_op = tf.check_numerics(K_ridge_op, "K_ridge_op: ")

            self.vars['X_dists_h'] = X_dists
            self.vars['ridge_h'] = ridge_ph
            self.ops['K_op'] = K_op
            self.ops['K_ridge_op'] = K_ridge_op

            # Nodes for xy computation
            K = tf.placeholder(tf.float32, name='K')
            K_inv = tf.placeholder(tf.float32, name='K_inv')
            xy_ = tf.placeholder(tf.float32, name='xy_')
            yt_ = tf.placeholder(tf.float32, name='yt_')
            K_inv_op = tf.matrix_inverse(K)
            if self.check_numerics:
                K_inv_op = tf.check_numerics(K_inv_op, "K_inv: ")
            xy_op = tf.matmul(K_inv, yt_)
            if self.check_numerics:
                xy_op = tf.check_numerics(xy_op, "xy_: ")

            self.vars['K_h'] = K
            self.vars['K_inv_h'] = K_inv
            self.vars['xy_h'] = xy_
            self.vars['yt_h'] = yt_
            self.ops['K_inv_op'] = K_inv_op
            self.ops['xy_op'] = xy_op

            # Nodes for yhat/sigma computation
            K2 = tf.placeholder(tf.float32, name="K2")
            K3 = tf.placeholder(tf.float32, name="K3")
            yhat_ = tf.cast(tf.matmul(tf.transpose(K2), xy_), tf.float32)
            if self.check_numerics:
                yhat_ = tf.check_numerics(yhat_, "yhat_: ")
            sv1 = tf.matmul(tf.transpose(K2), tf.matmul(K_inv, K2))
            if self.check_numerics:
                sv1 = tf.check_numerics(sv1, "sv1: ")
            sig_val = tf.cast((tf.sqrt(tf.diag_part(K3 - sv1))), tf.float32)
            if self.check_numerics:
                sig_val = tf.check_numerics(sig_val, "sig_val: ")

            self.vars['K2_h'] = K2
            self.vars['K3_h'] = K3
            self.ops['yhat_op'] = yhat_
            self.ops['sig_op'] = sig_val

            # Compute y_best (min y)
            y_best_op = tf.cast(tf.reduce_min(yt_, 0, True), tf.float32)
            if self.check_numerics:
                y_best_op = tf.check_numerics(y_best_op, "y_best_op: ")
            self.ops['y_best_op'] = y_best_op

            sigma = tf.placeholder(tf.float32, name='sigma')
            yhat = tf.placeholder(tf.float32, name='yhat')

            self.vars['sigma_h'] = sigma
            self.vars['yhat_h'] = yhat

    def __repr__(self):
        rep = ""
        for k, v in sorted(self.__dict__.items()):
            rep += "{} = {}\n".format(k, v)
        return rep

    def __str__(self):
        return self.__repr__()

    def check_X_y(self, X, y):
        from sklearn.utils.validation import check_X_y

        if X.shape[0] > self.max_train_size_:
            raise Exception("X_train size cannot exceed {} ({})"
                            .format(self.max_train_size_, X.shape[0]))
        return check_X_y(X, y, multi_output=True,
                         allow_nd=True, y_numeric=True,
                         estimator="GPR")

    def check_fitted(self):
        if self.X_train is None or self.y_train is None \
                or self.xy_ is None or self.K is None:
            raise Exception("The model must be trained before making predictions!")

    @staticmethod
    def check_array(X):
        from sklearn.utils.validation import check_array
        return check_array(X, allow_nd=True, estimator="GPR")

    @staticmethod
    def check_output(X):
        finite_els = np.isfinite(X)
        if not np.all(finite_els):
            raise Exception("Input contains non-finite values: {}"
                            .format(X[~finite_els]))

    def fit(self, X_train, y_train, ridge=1.0):
        self._reset()
        X_train, y_train = self.check_X_y(X_train, y_train)
        self.X_train = np.float32(X_train)
        self.y_train = np.float32(y_train)
        sample_size = self.X_train.shape[0]

        if np.isscalar(ridge):
            ridge = np.ones(sample_size) * ridge
        assert isinstance(ridge, np.ndarray)
        assert ridge.ndim == 1

        X_dists = np.zeros((sample_size, sample_size), dtype=np.float32)
        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(
                            intra_op_parallelism_threads=self.num_threads_)) as sess:
            dist_op = self.ops['dist_op']
            v1, v2 = self.vars['v1_h'], self.vars['v2_h']
            for i in range(sample_size):
                X_dists[i] = sess.run(dist_op, feed_dict={v1: self.X_train[i], v2: self.X_train})

            K_ridge_op = self.ops['K_ridge_op']
            X_dists_ph = self.vars['X_dists_h']
            ridge_ph = self.vars['ridge_h']

            self.K = sess.run(K_ridge_op, feed_dict={X_dists_ph: X_dists, ridge_ph: ridge})

            K_ph = self.vars['K_h']

            K_inv_op = self.ops['K_inv_op']
            self.K_inv = sess.run(K_inv_op, feed_dict={K_ph: self.K})

            xy_op = self.ops['xy_op']
            K_inv_ph = self.vars['K_inv_h']
            yt_ph = self.vars['yt_h']
            self.xy_ = sess.run(xy_op, feed_dict={K_inv_ph: self.K_inv,
                                                  yt_ph: self.y_train})
        return self

    def predict(self, X_test):
        self.check_fitted()
        X_test = np.float32(GPR.check_array(X_test))
        test_size = X_test.shape[0]
        sample_size = self.X_train.shape[0]

        arr_offset = 0
        yhats = np.zeros([test_size, 1])
        sigmas = np.zeros([test_size, 1])
        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(
                            intra_op_parallelism_threads=self.num_threads_)) as sess:
            # Nodes for distance operation
            dist_op = self.ops['dist_op']
            v1 = self.vars['v1_h']
            v2 = self.vars['v2_h']

            # Nodes for kernel computation
            K_op = self.ops['K_op']
            X_dists = self.vars['X_dists_h']

            # Nodes to compute yhats/sigmas
            yhat_ = self.ops['yhat_op']
            K_inv_ph = self.vars['K_inv_h']
            K2 = self.vars['K2_h']
            K3 = self.vars['K3_h']
            xy_ph = self.vars['xy_h']

            while arr_offset < test_size:
                if arr_offset + self.batch_size_ > test_size:
                    end_offset = test_size
                else:
                    end_offset = arr_offset + self.batch_size_

                X_test_batch = X_test[arr_offset:end_offset]
                batch_len = end_offset - arr_offset

                dists1 = np.zeros([sample_size, batch_len])
                for i in range(sample_size):
                    dists1[i] = sess.run(dist_op, feed_dict={v1: self.X_train[i],
                                                             v2: X_test_batch})

                sig_val = self.ops['sig_op']
                K2_ = sess.run(K_op, feed_dict={X_dists: dists1})
                yhat = sess.run(yhat_, feed_dict={K2: K2_, xy_ph: self.xy_})
                dists2 = np.zeros([batch_len, batch_len])
                for i in range(batch_len):
                    dists2[i] = sess.run(dist_op, feed_dict={v1: X_test_batch[i], v2: X_test_batch})
                K3_ = sess.run(K_op, feed_dict={X_dists: dists2})

                sigma = np.zeros([1, batch_len], np.float32)
                sigma[0] = sess.run(sig_val, feed_dict={K_inv_ph: self.K_inv, K2: K2_, K3: K3_})
                sigma = np.transpose(sigma)
                yhats[arr_offset: end_offset] = yhat
                sigmas[arr_offset: end_offset] = sigma
                arr_offset = end_offset
        GPR.check_output(yhats)
        GPR.check_output(sigmas)
        return GPRResult(yhats, sigmas)

    def get_params(self, deep=True):
        return {"length_scale": self.length_scale,
                "magnitude": self.magnitude,
                "X_train": self.X_train,
                "y_train": self.y_train,
                "xy_": self.xy_,
                "K": self.K,
                "K_inv": self.K_inv}

    def set_params(self, **parameters):
        for param, val in list(parameters.items()):
            setattr(self, param, val)
        return self

    def _reset(self):
        self.X_train = None
        self.y_train = None
        self.xy_ = None
        self.K = None
        self.K_inv = None
        self.graph = None
        self.build_graph()
        gc.collect()


class GPRGD(GPR):

    GP_BETA_UCB = "UCB"
    GP_BETA_CONST = "CONST"

    def __init__(self,
                 length_scale=1.0,
                 magnitude=1.0,
                 max_train_size=7000,
                 batch_size=3000,
                 num_threads=4,
                 learning_rate=0.01,
                 epsilon=1e-6,
                 max_iter=100,
                 sigma_multiplier=3.0,
                 mu_multiplier=1.0):
        super(GPRGD, self).__init__(length_scale=length_scale,
                                    magnitude=magnitude,
                                    max_train_size=max_train_size,
                                    batch_size=batch_size,
                                    num_threads=num_threads)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.sigma_multiplier = sigma_multiplier
        self.mu_multiplier = mu_multiplier
        self.X_min = None
        self.X_max = None

    def fit(self, X_train, y_train, X_min, X_max, ridge):  # pylint: disable=arguments-differ
        super(GPRGD, self).fit(X_train, y_train, ridge)
        self.X_min = X_min
        self.X_max = X_max

        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(
                            intra_op_parallelism_threads=self.num_threads_)) as sess:
            xt_ = tf.Variable(self.X_train[0], tf.float32)
            xt_ph = tf.placeholder(tf.float32)
            xt_assign_op = xt_.assign(xt_ph)
            init = tf.global_variables_initializer()
            sess.run(init)
            K2_mat = tf.transpose(tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.pow(
                tf.subtract(xt_, self.X_train), 2), 1)), 0))
            if self.check_numerics is True:
                K2_mat = tf.check_numerics(K2_mat, "K2_mat: ")
            K2__ = tf.cast(self.magnitude * tf.exp(-K2_mat / self.length_scale), tf.float32)
            if self.check_numerics is True:
                K2__ = tf.check_numerics(K2__, "K2__: ")
            yhat_gd = tf.cast(tf.matmul(tf.transpose(K2__), self.xy_), tf.float32)
            if self.check_numerics is True:
                yhat_gd = tf.check_numerics(yhat_gd, message="yhat: ")
            sig_val = tf.cast((tf.sqrt(self.magnitude - tf.matmul(
                tf.transpose(K2__), tf.matmul(self.K_inv, K2__)))), tf.float32)
            if self.check_numerics is True:
                sig_val = tf.check_numerics(sig_val, message="sigma: ")
            LOG.debug("\nyhat_gd : %s", str(sess.run(yhat_gd)))
            LOG.debug("\nsig_val : %s", str(sess.run(sig_val)))

            loss = tf.squeeze(tf.subtract(self.mu_multiplier * yhat_gd,
                                          self.sigma_multiplier * sig_val))
            if self.check_numerics is True:
                loss = tf.check_numerics(loss, "loss: ")
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                               epsilon=self.epsilon)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            train = optimizer.minimize(loss)

            self.vars['xt_'] = xt_
            self.vars['xt_ph'] = xt_ph
            self.ops['xt_assign_op'] = xt_assign_op
            self.ops['yhat_gd'] = yhat_gd
            self.ops['sig_val2'] = sig_val
            self.ops['loss_op'] = loss
            self.ops['train_op'] = train
        return self

    def predict(self, X_test, constraint_helper=None,  # pylint: disable=arguments-differ
                categorical_feature_method='hillclimbing',
                categorical_feature_steps=3):
        self.check_fitted()
        X_test = np.float32(GPR.check_array(X_test))
        test_size = X_test.shape[0]
        nfeats = self.X_train.shape[1]

        arr_offset = 0
        yhats = np.zeros([test_size, 1])
        sigmas = np.zeros([test_size, 1])
        minls = np.zeros([test_size, 1])
        minl_confs = np.zeros([test_size, nfeats])

        with tf.Session(graph=self.graph,
                        config=tf.ConfigProto(
                            intra_op_parallelism_threads=self.num_threads_)) as sess:
            while arr_offset < test_size:
                if arr_offset + self.batch_size_ > test_size:
                    end_offset = test_size
                else:
                    end_offset = arr_offset + self.batch_size_

                X_test_batch = X_test[arr_offset:end_offset]
                batch_len = end_offset - arr_offset

                xt_ = self.vars['xt_']
                init = tf.global_variables_initializer()
                sess.run(init)

                sig_val = self.ops['sig_val2']
                yhat_gd = self.ops['yhat_gd']
                loss = self.ops['loss_op']
                train = self.ops['train_op']

                xt_ph = self.vars['xt_ph']
                assign_op = self.ops['xt_assign_op']

                yhat = np.empty((batch_len, 1))
                sigma = np.empty((batch_len, 1))
                minl = np.empty((batch_len, 1))
                minl_conf = np.empty((batch_len, nfeats))
                for i in range(batch_len):
                    if self.debug is True:
                        LOG.info("-------------------------------------------")
                    yhats_it = np.empty((self.max_iter + 1,)) * np.nan
                    sigmas_it = np.empty((self.max_iter + 1,)) * np.nan
                    losses_it = np.empty((self.max_iter + 1,)) * np.nan
                    confs_it = np.empty((self.max_iter + 1, nfeats)) * np.nan

                    sess.run(assign_op, feed_dict={xt_ph: X_test_batch[i]})
                    step = 0
                    for step in range(self.max_iter):
                        if self.debug is True:
                            LOG.info("Batch %d, iter %d:", i, step)
                        yhats_it[step] = sess.run(yhat_gd)[0][0]
                        sigmas_it[step] = sess.run(sig_val)[0][0]
                        losses_it[step] = sess.run(loss)
                        confs_it[step] = sess.run(xt_)
                        if self.debug is True:
                            LOG.info("    yhat:  %s", str(yhats_it[step]))
                            LOG.info("    sigma: %s", str(sigmas_it[step]))
                            LOG.info("    loss:  %s", str(losses_it[step]))
                            LOG.info("    conf:  %s", str(confs_it[step]))
                        sess.run(train)
                        # constraint Projected Gradient Descent
                        xt = sess.run(xt_)
                        xt_valid = np.minimum(xt, self.X_max)
                        xt_valid = np.maximum(xt_valid, self.X_min)
                        sess.run(assign_op, feed_dict={xt_ph: xt_valid})
                        if constraint_helper is not None:
                            xt_valid = constraint_helper.apply_constraints(sess.run(xt_))
                            sess.run(assign_op, feed_dict={xt_ph: xt_valid})
                            if categorical_feature_method == 'hillclimbing':
                                if step % categorical_feature_steps == 0:
                                    current_xt = sess.run(xt_)
                                    current_loss = sess.run(loss)
                                    new_xt = \
                                        constraint_helper.randomize_categorical_features(
                                            current_xt)
                                    sess.run(assign_op, feed_dict={xt_ph: new_xt})
                                    new_loss = sess.run(loss)
                                    if current_loss < new_loss:
                                        sess.run(assign_op, feed_dict={xt_ph: new_xt})
                            else:
                                raise Exception("Unknown categorial feature method: {}".format(
                                    categorical_feature_method))
                    if step == self.max_iter - 1:
                        # Record results from final iteration
                        yhats_it[-1] = sess.run(yhat_gd)[0][0]
                        sigmas_it[-1] = sess.run(sig_val)[0][0]
                        losses_it[-1] = sess.run(loss)
                        confs_it[-1] = sess.run(xt_)
                        assert np.all(np.isfinite(yhats_it))
                        assert np.all(np.isfinite(sigmas_it))
                        assert np.all(np.isfinite(losses_it))
                        assert np.all(np.isfinite(confs_it))

                    # Store info for conf with min loss from all iters
                    if np.all(~np.isfinite(losses_it)):
                        min_loss_idx = 0
                    else:
                        min_loss_idx = np.nanargmin(losses_it)
                    yhat[i] = yhats_it[min_loss_idx]
                    sigma[i] = sigmas_it[min_loss_idx]
                    minl[i] = losses_it[min_loss_idx]
                    minl_conf[i] = confs_it[min_loss_idx]

                minls[arr_offset:end_offset] = minl
                minl_confs[arr_offset:end_offset] = minl_conf
                yhats[arr_offset:end_offset] = yhat
                sigmas[arr_offset:end_offset] = sigma
                arr_offset = end_offset

        GPR.check_output(yhats)
        GPR.check_output(sigmas)
        GPR.check_output(minls)
        GPR.check_output(minl_confs)

        return GPRGDResult(yhats, sigmas, minls, minl_confs)

    @staticmethod
    def calculate_sigma_multiplier(t, ndim, bound=0.1):
        assert t > 0
        assert ndim > 0
        assert bound > 0 and bound <= 1
        beta = 2 * np.log(ndim * (t**2) * (np.pi**2) / 6 * bound)
        if beta > 0:
            beta = np.sqrt(beta)
        else:
            beta = 1
        return beta


# def gp_tf(X_train, y_train, X_test, ridge, length_scale, magnitude, batch_size=3000):
#    with tf.Graph().as_default():
#        y_best = tf.cast(tf.reduce_min(y_train, 0, True), tf.float32)
#        sample_size = X_train.shape[0]
#        train_size = X_test.shape[0]
#        arr_offset = 0
#        yhats = np.zeros([train_size, 1])
#        sigmas = np.zeros([train_size, 1])
#        eips = np.zeros([train_size, 1])
#        X_train = np.float32(X_train)
#        y_train = np.float32(y_train)
#        X_test = np.float32(X_test)
#        ridge = np.float32(ridge)
#
#        v1 = tf.placeholder(tf.float32,name="v1")
#        v2 = tf.placeholder(tf.float32,name="v2")
#        dist_op = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2), 1))
#        try:
#            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
#
#            dists = np.zeros([sample_size,sample_size])
#            for i in range(sample_size):
#                dists[i] = sess.run(dist_op,feed_dict={v1:X_train[i], v2:X_train})
#
#
#            dists = tf.cast(dists, tf.float32)
#            K = magnitude * tf.exp(-dists/length_scale) + tf.diag(ridge);
#
#            K2 = tf.placeholder(tf.float32, name="K2")
#            K3 = tf.placeholder(tf.float32, name="K3")
#
#            x = tf.matmul(tf.matrix_inverse(K), y_train)
#            yhat_ =  tf.cast(tf.matmul(tf.transpose(K2), x), tf.float32);
#            sig_val = tf.cast((tf.sqrt(tf.diag_part(K3 -  tf.matmul(tf.transpose(K2),
#                                                                    tf.matmul(tf.matrix_inverse(K),
#                                                                              K2))))),
#                              tf.float32)
#
#            u = tf.placeholder(tf.float32, name="u")
#            phi1 = 0.5 * tf.erf(u / np.sqrt(2.0)) + 0.5
#            phi2 = (1.0 / np.sqrt(2.0 * np.pi)) * tf.exp(tf.square(u) * (-0.5));
#            eip = (tf.multiply(u, phi1) + phi2);
#
#            while arr_offset < train_size:
#                if arr_offset + batch_size > train_size:
#                    end_offset = train_size
#                else:
#                    end_offset = arr_offset + batch_size;
#
#                xt_ = X_test[arr_offset:end_offset];
#                batch_len = end_offset - arr_offset
#
#                dists = np.zeros([sample_size, batch_len])
#                for i in range(sample_size):
#                    dists[i] = sess.run(dist_op, feed_dict={v1:X_train[i], v2:xt_})
#
#                K2_ = magnitude * tf.exp(-dists / length_scale);
#                K2_ = sess.run(K2_)
#
#                dists = np.zeros([batch_len, batch_len])
#                for i in range(batch_len):
#                    dists[i] = sess.run(dist_op, feed_dict={v1:xt_[i], v2:xt_})
#                K3_ = magnitude * tf.exp(-dists / length_scale);
#                K3_ = sess.run(K3_)
#
#                yhat = sess.run(yhat_, feed_dict={K2:K2_})
#
#                sigma = np.zeros([1, batch_len], np.float32)
#                sigma[0] = (sess.run(sig_val, feed_dict={K2:K2_, K3:K3_}))
#                sigma = np.transpose(sigma)
#
#                u_ = tf.cast(tf.div(tf.subtract(y_best, yhat), sigma), tf.float32)
#                u_ = sess.run(u_)
#                eip_p = sess.run(eip, feed_dict={u:u_})
#                eip_ = tf.multiply(sigma, eip_p)
#                yhats[arr_offset:end_offset] = yhat
#                sigmas[arr_offset:end_offset] =  sigma;
#                eips[arr_offset:end_offset] = sess.run(eip_);
#                arr_offset = end_offset
#
#        finally:
#            sess.close()
#
#        return yhats, sigmas, eips


def euclidean_mat(X, y, sess):
    x_n = X.shape[0]
    y_n = y.shape[0]
    z = np.zeros([x_n, y_n])
    for i in range(x_n):
        v1 = X[i]
        tmp = []
        for j in range(y_n):
            v2 = y[j]
            tmp.append(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2))))
        z[i] = (sess.run(tmp))
    return z


def gd_tf(xs, ys, xt, ridge, length_scale=1.0, magnitude=1.0, max_iter=50):
    LOG.debug("xs shape: %s", str(xs.shape))
    LOG.debug("ys shape: %s", str(ys.shape))
    LOG.debug("xt shape: %s", str(xt.shape))
    with tf.Graph().as_default():
        # y_best = tf.cast(tf.reduce_min(ys,0,True),tf.float32);   #array
        # yhat_gd = tf.check_numerics(yhat_gd, message="yhat: ")
        sample_size = xs.shape[0]
        nfeats = xs.shape[1]
        test_size = xt.shape[0]
        # arr_offset = 0
        ini_size = xt.shape[0]

        yhats = np.zeros([test_size, 1])
        sigmas = np.zeros([test_size, 1])
        minl = np.zeros([test_size, 1])
        new_conf = np.zeros([test_size, nfeats])

        xs = np.float32(xs)
        ys = np.float32(ys)
        xt_ = tf.Variable(xt[0], tf.float32)

        sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
        init = tf.global_variables_initializer()
        sess.run(init)

        ridge = np.float32(ridge)
        v1 = tf.placeholder(tf.float32, name="v1")
        v2 = tf.placeholder(tf.float32, name="v2")
        dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2), 1))

        tmp = np.zeros([sample_size, sample_size])
        for i in range(sample_size):
            tmp[i] = sess.run(dist, feed_dict={v1: xs[i], v2: xs})

        tmp = tf.cast(tmp, tf.float32)
        K = magnitude * tf.exp(-tmp / length_scale) + tf.diag(ridge)
        LOG.debug("K shape: %s", str(sess.run(K).shape))

        K2_mat = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(xt_, xs), 2), 1))
        K2_mat = tf.transpose(tf.expand_dims(K2_mat, 0))
        K2 = tf.cast(tf.exp(-K2_mat / length_scale), tf.float32)

        x = tf.matmul(tf.matrix_inverse(K), ys)
        x = sess.run(x)
        yhat_ = tf.cast(tf.matmul(tf.transpose(K2), x), tf.float32)
        sig_val = tf.cast((tf.sqrt(magnitude - tf.matmul(
            tf.transpose(K2), tf.matmul(tf.matrix_inverse(K), K2)))), tf.float32)

        LOG.debug('yhat shape: %s', str(sess.run(yhat_).shape))
        LOG.debug('sig_val shape: %s', str(sess.run(sig_val).shape))
        yhat_ = tf.check_numerics(yhat_, message='yhat: ')
        sig_val = tf.check_numerics(sig_val, message='sig_val: ')
        loss = tf.squeeze(tf.subtract(yhat_, sig_val))
        loss = tf.check_numerics(loss, message='loss: ')
    #    optimizer = tf.train.GradientDescentOptimizer(0.1)
        LOG.debug('loss: %s', str(sess.run(loss)))
        optimizer = tf.train.AdamOptimizer(0.1)
        train = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(ini_size):
            assign_op = xt_.assign(xt[i])
            sess.run(assign_op)
            for step in range(max_iter):
                LOG.debug('sample #: %d, iter #: %d, loss: %s', i, step, str(sess.run(loss)))
                sess.run(train)
            yhats[i] = sess.run(yhat_)[0][0]
            sigmas[i] = sess.run(sig_val)[0][0]
            minl[i] = sess.run(loss)
            new_conf[i] = sess.run(xt_)
        return yhats, sigmas, minl, new_conf


def main():
    pass


def create_random_matrices(n_samples=3000, n_feats=12, n_test=4444):
    X_train = np.random.rand(n_samples, n_feats)
    y_train = np.random.rand(n_samples, 1)
    X_test = np.random.rand(n_test, n_feats)

    length_scale = np.random.rand()
    magnitude = np.random.rand()
    ridge = np.ones(n_samples) * np.random.rand()

    return X_train, y_train, X_test, length_scale, magnitude, ridge


if __name__ == "__main__":
    main()

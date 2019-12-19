#
# OtterTune - constants.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

# ---PIPELINE CONSTANTS---
#  how often to run the background tests, in seconds
RUN_EVERY = 300

#  the number of samples (staring points) in gradient descent
NUM_SAMPLES = 30

#  the number of selected tuning knobs
#  set it to a large value if you want to disable the knob identification
#  phase (i.e. tune all session knobs)
IMPORTANT_KNOB_NUMBER = 10000

#  top K config with best performance put into prediction
TOP_NUM_CONFIG = 10

# ---CONSTRAINTS CONSTANTS---

# address categorical knobs (enum, boolean)
ENABLE_DUMMY_ENCODER = False

#  Initial probability to flip categorical feature in apply_constraints
#  server/analysis/constraints.py
INIT_FLIP_PROB = 0.3

#  The probability that we flip the i_th categorical feature is
#  FLIP_PROB_DECAY * (probability we flip (i-1)_th categorical feature)
FLIP_PROB_DECAY = 0.5

# ---GPR CONSTANTS---
USE_GPFLOW = True

GPR_DEBUG = True

DEFAULT_LENGTH_SCALE = 2.0

DEFAULT_MAGNITUDE = 1.0

#  Max training size in GPR model
MAX_TRAIN_SIZE = 7000

#  Batch size in GPR model
BATCH_SIZE = 3000

#  Threads for TensorFlow config
NUM_THREADS = 4

#  Value of beta for UCB
UCB_BETA = 'get_beta_td'

#  Name of the GPR model to use (GPFLOW only)
GPR_MODEL_NAME = 'BasicGP'

# ---GRADIENT DESCENT CONSTANTS---
#  the maximum iterations of gradient descent
MAX_ITER = 500

DEFAULT_LEARNING_RATE = 0.01

# ---GRADIENT DESCENT FOR GPR---
#  a small bias when using training data points as starting points.
GPR_EPS = 0.001

DEFAULT_RIDGE = 1.00

DEFAULT_EPSILON = 1e-6

DEFAULT_SIGMA_MULTIPLIER = 1.0

DEFAULT_MU_MULTIPLIER = 1.0

DEFAULT_UCB_SCALE = 0.2

# ---HYPERPARAMETER TUNING FOR GPR---
HP_MAX_ITER = 5000

HP_LEARNING_RATE = 0.001

# ---GRADIENT DESCENT FOR DNN---
DNN_TRAIN_ITER = 100

# Gradient Descent iteration for recommendation
DNN_GD_ITER = 100

DNN_EXPLORE = False

DNN_EXPLORE_ITER = 500

# noise scale for paramater space exploration
DNN_NOISE_SCALE_BEGIN = 0.1

DNN_NOISE_SCALE_END = 0.0

DNN_DEBUG = True

DNN_DEBUG_INTERVAL = 100

# ---DDPG CONSTRAINTS CONSTANTS---

#  Use a simple reward
DDPG_SIMPLE_REWARD = True

#  The weight of future rewards in Q value
DDPG_GAMMA = 0.0

#  Batch size in DDPG model
DDPG_BATCH_SIZE = 32

#  Learning rate of actor network
ACTOR_LEARNING_RATE = 0.02

#  Learning rate of critic network
CRITIC_LEARNING_RATE = 0.001

#  Number of update epochs per iteration
UPDATE_EPOCHS = 30

#  The number of hidden units in each layer of the actor MLP
ACTOR_HIDDEN_SIZES = [128, 128, 64]

#  The number of hidden units in each layer of the critic MLP
CRITIC_HIDDEN_SIZES = [64, 128, 64]

#  Use the same setting from the CDBTune paper
USE_DEFAULT = False
#  Overwrite the DDPG settings if using CDBTune
if USE_DEFAULT:
    DDPG_SIMPLE_REWARD = False
    DDPG_GAMMA = 0.99
    DDPG_BATCH_SIZE = 32
    ACTOR_LEARNING_RATE = 0.001
    CRITIC_LEARNING_RATE = 0.001
    UPDATE_EPOCHS = 1

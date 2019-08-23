#
# OtterTune - constants.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

# ---PIPELINE CONSTANTS---
#  the number of samples (staring points) in gradient descent
NUM_SAMPLES = 30

#  the number of selected tuning knobs
IMPORTANT_KNOB_NUMBER = 10

#  top K config with best performance put into prediction
TOP_NUM_CONFIG = 10

# ---CONSTRAINTS CONSTANTS---

#  Initial probability to flip categorical feature in apply_constraints
#  server/analysis/constraints.py
INIT_FLIP_PROB = 0.3

#  The probability that we flip the i_th categorical feature is
#  FLIP_PROB_DECAY * (probability we flip (i-1)_th categorical feature)
FLIP_PROB_DECAY = 0.5

# ---GPR CONSTANTS---
DEFAULT_LENGTH_SCALE = 1.0

DEFAULT_MAGNITUDE = 1.0

#  Max training size in GPR model
MAX_TRAIN_SIZE = 7000

#  Batch size in GPR model
BATCH_SIZE = 3000

# Threads for TensorFlow config
NUM_THREADS = 4

# ---GRADIENT DESCENT CONSTANTS---
#  the maximum iterations of gradient descent
MAX_ITER = 500

#  a small bias when using training data points as starting points.
GPR_EPS = 0.001

DEFAULT_RIDGE = 0.01

DEFAULT_LEARNING_RATE = 0.01

DEFAULT_EPSILON = 1e-6

DEFAULT_SIGMA_MULTIPLIER = 3.0

DEFAULT_MU_MULTIPLIER = 1.0

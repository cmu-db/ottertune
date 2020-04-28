#
# OtterTune - constants.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from website.types import DBMSType

# These parameters are not specified for any session, so they can only be set here

# If this flag is set, we check if celery is running, and restart celery if it is not.
CHECK_CELERY = True

# address categorical knobs (enum, boolean)
ENABLE_DUMMY_ENCODER = False

# Whether to include the pruned metrics from the workload characterization subtask in
# the output (y) when ranking the knobs for a given workload in the knob identification
# subtask.

# When computing the ranked knobs in the knob identification subtask, the output (y) is
# the set of target objectives used to tune the given workload. If this flag is enabled
# then the pruned metrics from the workload characterization subtask are also included
# in the output. (See website/tasks/periodic_tasks.py)
KNOB_IDENT_USE_PRUNED_METRICS = False

# The background tasks only process workloads containing this minimum amount of results
MIN_WORKLOAD_RESULTS_COUNT = 5

# The views used for metrics pruning
VIEWS_FOR_PRUNING = {
    DBMSType.ORACLE: ['dba_hist_osstat', 'dba_hist_sysstat', 'dba_hist_system_event',
                      'dba_workload_replays', 'dba_hist_sys_time_model'],
}

# The views used for DDPG
# WARNING: modifying this parameter will cause all existing DDPG sessions broken
VIEWS_FOR_DDPG = {
    DBMSType.ORACLE: ['dba_hist_sys_time_model'],
}

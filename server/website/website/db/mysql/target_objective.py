#
# OtterTune - target_objective.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from website.types import DBMSType
from ..base.target_objective import BaseThroughput  # pylint: disable=relative-beyond-top-level

target_objective_list = tuple((DBMSType.MYSQL, target_obj) for target_obj in [  # pylint: disable=invalid-name
    BaseThroughput(transactions_counter=('innodb_metrics.trx_rw_commits',
                                         'innodb_metrics.trx_ro_commits',
                                         'innodb_metrics.trx_nl_ro_commits'))
])

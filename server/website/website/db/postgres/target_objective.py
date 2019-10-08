#
# OtterTune - target_objective.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from ..base.target_objective import BaseThroughput
from website.types import DBMSType

target_objective_list = tuple((DBMSType.POSTGRES, target_obj) for target_obj in [  # pylint: disable=invalid-name
    BaseThroughput(transactions_counter='pg_stat_database.xact_commit')
])

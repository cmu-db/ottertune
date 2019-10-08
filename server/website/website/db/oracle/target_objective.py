#
# OtterTune - target_objective.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from ..base.target_objective import BaseTargetObjective, BaseThroughput, LESS_IS_BETTER
from website.types import DBMSType


class DBTime(BaseTargetObjective):

    def __init__(self):
        super().__init__(name='db_time', pprint='DB Time', unit='milliseconds', short_unit='ms',
                         improvement=LESS_IS_BETTER)

    def compute(self, metrics, observation_time):
        metric_names = ('global.db cpu', 'global.cursor: pin s wait on x.time_waited',
                        'global.user i/o wait time')
        db_time = float(sum(metrics[mname] for mname in metric_names)) / observation_time
        return db_time


target_objective_list = tuple((DBMSType.ORACLE, target_obj) for target_obj in [  # pylint: disable=invalid-name
    BaseThroughput(transactions_counter='global.user commits'),
    DBTime(),
])

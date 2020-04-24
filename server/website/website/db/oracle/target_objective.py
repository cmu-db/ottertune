#
# OtterTune - target_objective.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

import logging

from website.models import DBMSCatalog, MetricCatalog
from website.types import DBMSType
from ..base.target_objective import (BaseTargetObjective, BaseThroughput, LESS_IS_BETTER,
                                     MORE_IS_BETTER)

LOG = logging.getLogger(__name__)


class CustomDBTime(BaseTargetObjective):

    def __init__(self):
        super().__init__(name='custom_db_time', pprint='Custom DB Time', unit='seconds',
                         short_unit='s', improvement=LESS_IS_BETTER)

    def compute(self, metrics, observation_time):
        total_wait_time = 0.
        # dba_hist db_time will be 0 after cleaning if & only if it does not exist before cleaning
        has_dba_hist = metrics['global.dba_hist_sys_time_model.db time'] > 0
        for name, value in metrics.items():
            if has_dba_hist and 'dba_hist_' not in name:
                continue
            if 'db cpu' in name:
                total_wait_time += float(value)
            elif 'time_waited_micro_fg' in name:
                wait_time = float(value)
            elif name.endswith('wait_class'):
                # wait_class#:
                # 0: Other; 1: Application; 2: Configuration; 3: Administrative; 4: Concurrency;
                # 5: Commit; 6: Idle; 7: Network; 8: User I/O; 9: System I/O
                if value == 'Idle':
                    wait_time = 0
                total_wait_time += wait_time
        return total_wait_time / 1000000.


class NormalizedDBTime(BaseTargetObjective):

    def __init__(self):
        super().__init__(name='db_time', pprint='Normalized DB Time', unit='seconds',
                         short_unit='s', improvement=LESS_IS_BETTER)
        # This target objective is designed for Oracle v12.2.0.1.0
        dbms = DBMSCatalog.objects.get(type=DBMSType.ORACLE, version='12.2.0.1.0')
        self.default_values = {}
        for metric in MetricCatalog.objects.filter(dbms=dbms):
            self.default_values[metric.name] = metric.default

    def reload_default_metrics(self):
        dbms = DBMSCatalog.objects.get(type=DBMSType.ORACLE, version='12.2.0.1.0')
        self.default_values = {}
        for metric in MetricCatalog.objects.filter(dbms=dbms):
            self.default_values[metric.name] = metric.default

    def compute(self, metrics, observation_time):
        extra_io_metrics = ["log file sync"]
        not_io_metrics = ["read by other session"]
        total_wait_time = 0.
        has_dba_hist = metrics['global.dba_hist_sys_time_model.db time'] > 0
        for name, value in metrics.items():
            if has_dba_hist and 'dba_hist_' not in name:
                continue
            if 'db cpu' in name:
                total_wait_time += float(value)
            elif 'time_waited_micro_fg' in name:
                default_wait_time = float(self.default_values[name])
                wait_time = float(value)
            elif 'total_waits_fg' in name:
                default_total_waits = float(self.default_values[name])
                total_waits = float(value)
            elif name.endswith('wait_class'):
                if value == 'Idle':
                    wait_time = 0
                elif value in ('User I/O', 'System I/O') or \
                        any(n in name for n in extra_io_metrics):
                    if not any(n in name for n in not_io_metrics):
                        if default_total_waits == 0:
                            average_wait = 0
                        else:
                            average_wait = default_wait_time / default_total_waits
                        wait_time = total_waits * average_wait
                total_wait_time += wait_time
        return total_wait_time / 1000000.


class RawDBTime(BaseTargetObjective):

    def __init__(self):
        super().__init__(name='raw_db_time', pprint='Raw DB Time',
                         unit='seconds', short_unit='s', improvement=LESS_IS_BETTER)

    def compute(self, metrics, observation_time):
        has_dba_hist = metrics['global.dba_hist_sys_time_model.db time'] > 0
        if has_dba_hist:
            return metrics['global.dba_hist_sys_time_model.db time'] / 1000000.
        return metrics['global.sys_time_model.db time'] / 1000000.


class TransactionCounter(BaseTargetObjective):

    def __init__(self):
        super().__init__(name='transaction_counter', pprint='Number of commits and rollbacks',
                         unit='transactions', short_unit='txn', improvement=MORE_IS_BETTER)

    def compute(self, metrics, observation_time):
        num_txns = sum(metrics[ctr] for ctr in ('global.sysstat.user commits',
                                                'global.sysstat.user rollbacks'))
        return num_txns


class ElapsedTime(BaseTargetObjective):

    def __init__(self):
        super().__init__(name='elapsed_time', pprint='Elapsed Time', unit='seconds',
                         short_unit='s', improvement=LESS_IS_BETTER)

    def compute(self, metrics, observation_time):
        return observation_time


target_objective_list = tuple((DBMSType.ORACLE, target_obj) for target_obj in [  # pylint: disable=invalid-name
    BaseThroughput(transactions_counter=('global.sysstat.user commits',
                                         'global.sysstat.user rollbacks')),
    CustomDBTime(),
    NormalizedDBTime(),
    RawDBTime(),
    TransactionCounter(),
    ElapsedTime(),
])

#
# OtterTune - target_objective.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

import copy
import logging
from collections import OrderedDict

from website import models, types

LOG = logging.getLogger(__name__)

# Direction of performance improvement
LESS_IS_BETTER = '(less is better)'
MORE_IS_BETTER = '(more is better)'
THROUGHPUT = 'throughput_txn_per_sec'


class BaseMetric:

    _improvement_choices = (LESS_IS_BETTER, MORE_IS_BETTER, '')

    def __init__(self, name, pprint=None, unit='events / second', short_unit='events/sec',
                 improvement='', scale=1):
        if improvement not in self._improvement_choices:
            raise ValueError("Improvement must be one of: {}".format(
                ', '.join("'{}'".format(ic) for ic in self._improvement_choices)))
        if scale != 1:
            raise NotImplementedError()

        self.name = name
        self.pprint = pprint or name
        self.unit = unit
        self.short_unit = short_unit
        self.improvement = improvement
        self.scale = scale


class BaseTargetObjective(BaseMetric):
    _improvement_choices = (LESS_IS_BETTER, MORE_IS_BETTER)

    def __init__(self, name, pprint, unit, short_unit, improvement, scale=1):
        super().__init__(name=name, pprint=pprint, unit=unit, short_unit=short_unit,
                         improvement=improvement, scale=scale)

    @property
    def label(self):
        return '{} ({})'.format(self.pprint, self.short_unit)

    def compute(self, metrics, observation_time):
        raise NotImplementedError()


class BaseThroughput(BaseTargetObjective):

    def __init__(self, transactions_counter):
        super().__init__(name=THROUGHPUT, pprint='Throughput',
                         unit='transactions / second', short_unit='txn/sec',
                         improvement=MORE_IS_BETTER)
        self.transactions_counter = transactions_counter

    def compute(self, metrics, observation_time):
        return float(metrics[self.transactions_counter]) / observation_time


class TargetObjectives:
    LESS_IS_BETTER = LESS_IS_BETTER
    MORE_IS_BETTER = MORE_IS_BETTER
    THROUGHPUT = THROUGHPUT

    def __init__(self):
        self._registry = {}
        self._metric_metadatas = {}
        self._default_target_objective = THROUGHPUT

    def register(self):
        from ..myrocks.target_objective import target_objective_list as _myrocks_list
        from ..oracle.target_objective import target_objective_list as _oracle_list
        from ..postgres.target_objective import target_objective_list as _postgres_list

        if not self.registered():
            LOG.info('Registering target objectives...')
            full_list = _myrocks_list + _oracle_list + _postgres_list
            for dbms_type, target_objective_instance in full_list:
                dbmss = models.DBMSCatalog.objects.filter(type=dbms_type)
                name = target_objective_instance.name

                for dbms in dbmss:
                    dbms_id = int(dbms.pk)
                    if dbms_id not in self._registry:
                        self._registry[dbms_id] = {}
                    self._registry[dbms_id][name] = target_objective_instance

                    if dbms_id not in self._metric_metadatas:
                        numeric_metrics = models.MetricCatalog.objects.filter(dbms=dbms).exclude(
                            metric_type=types.MetricType.INFO).values_list('name', flat=True)
                        self._metric_metadatas[dbms_id] = [(mname, BaseMetric(mname)) for mname
                                                           in sorted(numeric_metrics)]

    def registered(self):
        return len(self._registry) > 0

    def get_metric_metadata(self, dbms_id, target_objective):
        if not self.registered():
            self.register()
        dbms_id = int(dbms_id)
        metadata = list(self._metric_metadatas[dbms_id])
        target_objective_instance = self._registry[dbms_id][target_objective]
        metadata.insert(0, (target_objective, target_objective_instance))
        return OrderedDict(metadata)

    def default(self):
        return self._default_target_objective

    def get_instance(self, dbms_id, target_objective):
        if not self.registered():
            self.register()
        dbms_id = int(dbms_id)
        instance = self._registry[dbms_id][target_objective]
        return instance

    def get_all(self, dbms_id=None):
        if not self.registered():
            self.register()
        if dbms_id is None:
            res = copy.deepcopy(self._registry)
        else:
            dbms_id = int(dbms_id)
            res = copy.deepcopy(self._registry[dbms_id])
        return res

    def __repr__(self):
        s = 'TargetObjectives = (\n'
        for dbms_id, entry in self._registry.items():  # pylint: disable=not-an-iterable
            s += '  {}:\n'.format(models.DBMSCatalog.objects.get(id=dbms_id).full_name)
            for name in entry.keys():
                s += '    {}\n'.format(name)
        s += ')\n'
        return s


target_objectives = TargetObjectives()  # pylint: disable=invalid-name

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

    def is_udf(self):  # pylint: disable=no-self-use
        return False


class BaseThroughput(BaseTargetObjective):

    def __init__(self, transactions_counter):
        super().__init__(name=THROUGHPUT, pprint='Throughput',
                         unit='transactions / second', short_unit='txn/sec',
                         improvement=MORE_IS_BETTER)
        if not isinstance(transactions_counter, (str, tuple)):
            raise TypeError(
                "Argument 'transactions_counter' must be str or tuple type, not {}.".format(
                    type(transactions_counter)))
        self.transactions_counter = transactions_counter

    def compute(self, metrics, observation_time):
        if isinstance(self.transactions_counter, tuple):
            num_txns = sum(metrics[ctr] for ctr in self.transactions_counter)
        else:
            num_txns = metrics[self.transactions_counter]
        return float(num_txns) / observation_time


class BaseUserDefinedTarget(BaseTargetObjective):
    _improvement_choices = (LESS_IS_BETTER, MORE_IS_BETTER, '')

    def __init__(self, target_name, improvement, unit='unknown', short_unit='unknown', pprint=None):
        if pprint is None:
            pprint = 'udf.' + target_name
        super().__init__(name=target_name, pprint=pprint, unit=unit,
                         short_unit=short_unit, improvement=improvement)

    def is_udf(self):
        return True

    def compute(self, metrics, observation_time):
        name = 'udm.' + self.name
        if name not in metrics:
            LOG.warning('cannot find the user defined target objective %s,\
                        return 0 instead', self.name)
        return metrics.get(name, 0)


class TargetObjectives:
    LESS_IS_BETTER = LESS_IS_BETTER
    MORE_IS_BETTER = MORE_IS_BETTER
    THROUGHPUT = THROUGHPUT

    def __init__(self):
        self._registry = {}
        self._metric_metadatas = {}
        self._udm_metadatas = {}  # user defined metrics
        self._default_target_objective = THROUGHPUT

    def register(self):
        from ..myrocks.target_objective import target_objective_list as _myrocks_list  # pylint: disable=import-outside-toplevel
        from ..oracle.target_objective import target_objective_list as _oracle_list  # pylint: disable=import-outside-toplevel
        from ..postgres.target_objective import target_objective_list as _postgres_list  # pylint: disable=import-outside-toplevel
        from ..mysql.target_objective import target_objective_list as _mysql_list  # pylint: disable=import-outside-toplevel

        if not self.registered():
            LOG.info('Registering target objectives...')
            full_list = _myrocks_list + _oracle_list + _postgres_list + _mysql_list
            for dbms_type, target_objective_instance in full_list:
                dbmss = models.DBMSCatalog.objects.filter(type=dbms_type)
                name = target_objective_instance.name

                for dbms in dbmss:
                    dbms_id = int(dbms.pk)
                    if dbms_id not in self._registry:
                        self._registry[dbms_id] = {}
                    self._registry[dbms_id][name] = target_objective_instance

                    if dbms_id not in self._metric_metadatas:
                        numeric_metrics = models.MetricCatalog.objects.filter(
                            dbms=dbms, metric_type__in=types.MetricType.numeric()).values_list(
                                'name', flat=True)
                        self._metric_metadatas[dbms_id] = [(mname, BaseMetric(mname)) for mname
                                                           in sorted(numeric_metrics)]

    def registered(self):
        return len(self._registry) > 0

    def udm_registered(self, dbms_id):
        return dbms_id in self._udm_metadatas

    def register_udm(self, dbms_id, metrics):
        if dbms_id in self._udm_metadatas:
            LOG.warning('User Defined Metrics have already been registered, append to existing one')
            metadatas = self._udm_metadatas[dbms_id]
        else:
            metadatas = []
        for name, info in metrics.items():
            name = 'udm.' + name
            metadatas.append((name,
                              BaseMetric(name, unit=info['unit'], short_unit=info['short_unit'])))
        self._udm_metadatas[dbms_id] = metadatas

    def get_metric_metadata(self, dbms_id, target_objective):
        if not self.registered():
            self.register()
        dbms_id = int(dbms_id)
        targets_list = []
        for target_name, target_instance in self._registry[dbms_id].items():
            if target_name == target_objective:
                targets_list.insert(0, (target_name, target_instance))
            elif not target_instance.is_udf():
                targets_list.append((target_name, target_instance))
        if dbms_id in self._udm_metadatas:
            metadata = targets_list + list(self._udm_metadatas[dbms_id]) +\
                       list(self._metric_metadatas[dbms_id])
        else:
            metric_meta = list(self._metric_metadatas[dbms_id])
            udm_metric_meta = []
            db_metric_meta = []
            for metric_name, metric in metric_meta:
                if metric_name.startswith('udm.'):
                    udm_metric_meta.append((metric_name, metric))
                else:
                    db_metric_meta.append((metric_name, metric))
            metadata = targets_list + udm_metric_meta + db_metric_meta
        meta_dict = OrderedDict()
        for metric_name, metric in metadata:
            if metric_name not in meta_dict:
                meta_dict[metric_name] = metric
        return meta_dict

    def default(self):
        return self._default_target_objective

    def get_instance(self, dbms_id, target_objective):
        if not self.registered():
            self.register()
        dbms_id = int(dbms_id)
        instance = self._registry[dbms_id].get(target_objective, 'None')
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

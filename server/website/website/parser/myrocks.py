#
# OtterTune - myrocks.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Jan 16, 2018

@author: bohan
'''

import re
from collections import OrderedDict

from .base import BaseParser
from website.models import DBMSCatalog
from website.types import DBMSType, KnobUnitType, MetricType, VarType
from website.utils import ConversionUtil


class MyRocksParser(BaseParser):

    MYROCKS_BYTES_SYSTEM = [
        (1024 ** 5, 'PB'),
        (1024 ** 4, 'TB'),
        (1024 ** 3, 'GB'),
        (1024 ** 2, 'MB'),
        (1024 ** 1, 'kB'),
        (1024 ** 0, 'B'),
    ]

    MYROCKS_TIME_SYSTEM = [
        (1000 * 60 * 60 * 24, 'd'),
        (1000 * 60 * 60, 'h'),
        (1000 * 60, 'min'),
        (1, 'ms'),
        (1000, 's'),
    ]

    MYROCKS_BASE_KNOBS = {
        'session_variables.rocksdb_max_open_files': '-1'
    }

    @property
    def base_configuration_settings(self):
        return dict(self.MYROCKS_BASE_KNOBS)

    @property
    def knob_configuration_filename(self):
        return 'myrocks.conf'

    @property
    def transactions_counter(self):
        return 'session_status.questions'

    def latency_timer(self):
        return 'session_status.questions'

    def convert_integer(self, int_value, metadata):
        converted = None
        try:
            converted = super(MyRocksParser, self).convert_integer(
                int_value, metadata)
        except ValueError:
            if metadata.unit == KnobUnitType.BYTES:
                converted = ConversionUtil.get_raw_size(
                    int_value, system=self.MYROCKS_BYTES_SYSTEM)
            elif metadata.unit == KnobUnitType.MILLISECONDS:
                converted = ConversionUtil.get_raw_size(
                    int_value, system=self.MYROCKS_TIME_SYSTEM)
            else:
                raise Exception('Unknown unit type: {}'.format(metadata.unit))
        if converted is None:
            raise Exception('Invalid integer format for {}: {}'.format(
                metadata.name, int_value))
        return converted

    def format_integer(self, int_value, metadata):
        if metadata.unit != KnobUnitType.OTHER and int_value > 0:
            if metadata.unit == KnobUnitType.BYTES:
                int_value = ConversionUtil.get_human_readable(
                    int_value, MyRocksParser.MYROCKS_BYTES_SYSTEM)
            elif metadata.unit == KnobUnitType.MILLISECONDS:
                int_value = ConversionUtil.get_human_readable(
                    int_value, MyRocksParser.MYROCKS_TIME_SYSTEM)
            else:
                raise Exception('Invalid unit type for {}: {}'.format(
                    metadata.name, metadata.unit))
        else:
            int_value = super(MyRocksParser, self).format_integer(
                int_value, metadata)
        return int_value

    def parse_version_string(self, version_string):
        dbms_version = version_string.split(',')[0]
        return re.search(r'\d+\.\d+(?=\.\d+)', dbms_version).group(0)

    def parse_helper(self, scope, valid_variables, view_variables):
        for view_name, variables in list(view_variables.items()):
            if scope == 'local':
                for obj_name, sub_vars in list(variables.items()):
                    for var_name, var_value in list(sub_vars.items()):  # local
                        full_name = '{}.{}.{}'.format(view_name, obj_name, var_name)
                        valid_variables[full_name] = var_value
            elif scope == 'global':
                for var_name, var_value in list(variables.items()):  # global
                    full_name = '{}.{}'.format(view_name, var_name)
                    valid_variables[full_name] = var_value
            else:
                raise Exception('Unsupported variable scope: {}'.format(scope))
        return valid_variables

    # global variable fullname: viewname.varname
    # local variable fullname: viewname.objname.varname
    # return format: valid_variables = {var_fullname:var_val}
    def parse_dbms_variables(self, variables):
        valid_variables = {}
        for scope, sub_vars in list(variables.items()):
            if sub_vars is None:
                continue
            if scope == 'global':
                valid_variables.update(self.parse_helper('global', valid_variables, sub_vars))
            elif scope == 'local':
                for _, viewnames in list(sub_vars.items()):
                    for viewname, objnames in list(viewnames.items()):
                        for obj_name, view_vars in list(objnames.items()):
                            valid_variables.update(self.parse_helper(
                                'local', valid_variables, {viewname: {obj_name: view_vars}}))
            else:
                raise Exception('Unsupported variable scope: {}'.format(scope))
        return valid_variables

    # local variable: viewname.objname.varname
    # global variable: viewname.varname
    # This function is to change local variable fullname to viewname.varname, global
    # variable remains same. This is because local varialbe in knob_catalog is in
    # parial format (i,e. viewname.varname)
    @staticmethod
    def partial_name(full_name):
        var_name = full_name.split('.')
        if len(var_name) == 2:  # global variable
            return full_name
        elif len(var_name) == 3:  # local variable
            return var_name[0] + '.' + var_name[2]
        else:
            raise Exception('Invalid variable full name: {}'.format(full_name))

    @staticmethod
    def extract_valid_variables(variables, catalog, default_value=None):
        valid_variables = {}
        diff_log = []
        valid_lc_variables = {k.lower(): v for k, v in list(catalog.items())}

        # First check that the names of all variables are valid (i.e., listed
        # in the official catalog). Invalid variables are logged as 'extras'.
        # Variable names that are valid but differ in capitalization are still
        # added to valid_variables but with the proper capitalization. They
        # are also logged as 'miscapitalized'.
        for var_name, var_value in list(variables.items()):
            lc_var_name = var_name.lower()
            prt_name = MyRocksParser.partial_name(lc_var_name)
            if prt_name in valid_lc_variables:
                valid_name = valid_lc_variables[prt_name].name
                if prt_name != valid_name:
                    diff_log.append(('miscapitalized', valid_name, var_name, var_value))
                valid_variables[var_name] = var_value
            else:
                diff_log.append(('extra', None, var_name, var_value))

        # Next find all item names that are listed in the catalog but missing from
        # variables. Missing global variables are added to valid_variables with
        # the given default_value if provided (or the item's actual default value
        # if not) and logged as 'missing'. For now missing local variables are
        # not added to valid_variables
        lc_variables = {MyRocksParser.partial_name(k.lower()): v
                        for k, v in list(variables.items())}
        for valid_lc_name, metadata in list(valid_lc_variables.items()):
            if valid_lc_name not in lc_variables:
                diff_log.append(('missing', metadata.name, None, None))
                if metadata.scope == 'global':
                    valid_variables[metadata.name] = default_value if \
                        default_value is not None else metadata.default
        return valid_variables, diff_log

    def calculate_change_in_metrics(self, metrics_start, metrics_end):
        adjusted_metrics = {}
        for met_name, start_val in list(metrics_start.items()):
            end_val = metrics_end[met_name]
            met_info = self.metric_catalog_[MyRocksParser.partial_name(met_name)]
            if met_info.vartype == VarType.INTEGER or \
                    met_info.vartype == VarType.REAL:
                conversion_fn = self.convert_integer if \
                    met_info.vartype == VarType.INTEGER else \
                    self.convert_real
                start_val = conversion_fn(start_val, met_info)
                end_val = conversion_fn(end_val, met_info)
                adj_val = end_val - start_val
                assert adj_val >= 0
                adjusted_metrics[met_name] = adj_val
            else:
                # This metric is either a bool, enum, string, or timestamp
                # so take last recorded value from metrics_end
                adjusted_metrics[met_name] = end_val
        return adjusted_metrics

    def parse_dbms_knobs(self, knobs):
        valid_knobs = self.parse_dbms_variables(knobs)
        # Extract all valid knobs
        return MyRocksParser.extract_valid_variables(
            valid_knobs, self.knob_catalog_)

    def parse_dbms_metrics(self, metrics):
        valid_metrics = self.parse_dbms_variables(metrics)
        # Extract all valid metrics
        valid_metrics, diffs = MyRocksParser.extract_valid_variables(
            valid_metrics, self.metric_catalog_, default_value='0')
        return valid_metrics, diffs

    def convert_dbms_metrics(self, metrics, observation_time, target_objective=None):
        metric_data = {}
        for name, value in list(metrics.items()):
            prt_name = MyRocksParser.partial_name(name)
            if prt_name in self.numeric_metric_catalog_:
                metadata = self.numeric_metric_catalog_[prt_name]
                if metadata.metric_type == MetricType.COUNTER:
                    converted = self.convert_integer(value, metadata)
                    metric_data[name] = float(converted) / observation_time
                else:
                    raise Exception('Unknown metric type for {}: {}'.format(
                        name, metadata.metric_type))

        if target_objective is not None and self.target_metric(target_objective) not in metric_data:
            raise Exception("Cannot find objective function")

        if target_objective is not None:
            metric_data[target_objective] = metric_data[self.target_metric(target_objective)]
        else:
            # default
            metric_data['throughput_txn_per_sec'] = \
                metric_data[self.target_metric(target_objective)]
        return metric_data

    def convert_dbms_knobs(self, knobs):
        knob_data = {}
        for name, value in list(knobs.items()):
            prt_name = MyRocksParser.partial_name(name)
            if prt_name in self.tunable_knob_catalog_:
                metadata = self.tunable_knob_catalog_[prt_name]
                assert(metadata.tunable)
                value = knobs[name]
                conv_value = None
                if metadata.vartype == VarType.BOOL:
                    conv_value = self.convert_bool(value, metadata)
                elif metadata.vartype == VarType.ENUM:
                    conv_value = self.convert_enum(value, metadata)
                elif metadata.vartype == VarType.INTEGER:
                    conv_value = self.convert_integer(value, metadata)
                elif metadata.vartype == VarType.REAL:
                    conv_value = self.convert_real(value, metadata)
                elif metadata.vartype == VarType.STRING:
                    conv_value = self.convert_string(value, metadata)
                elif metadata.vartype == VarType.TIMESTAMP:
                    conv_value = self.convert_timestamp(value, metadata)
                else:
                    raise Exception(
                        'Unknown variable type: {}'.format(metadata.vartype))
                if conv_value is None:
                    raise Exception(
                        'Param value for {} cannot be null'.format(name))
                knob_data[name] = conv_value
        return knob_data

    def filter_numeric_metrics(self, metrics):
        return OrderedDict([(k, v) for k, v in list(metrics.items()) if
                            MyRocksParser.partial_name(k) in self.numeric_metric_catalog_])

    def filter_tunable_knobs(self, knobs):
        return OrderedDict([(k, v) for k, v in list(knobs.items()) if
                            MyRocksParser.partial_name(k) in self.tunable_knob_catalog_])


class MyRocks56Parser(MyRocksParser):

    def __init__(self):
        dbms = DBMSCatalog.objects.get(
            type=DBMSType.MYROCKS, version='5.6')
        super(MyRocks56Parser, self).__init__(dbms.pk)

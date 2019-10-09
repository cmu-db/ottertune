#
# OtterTune - parser.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

import re
from collections import OrderedDict

from ..base.parser import BaseParser
from .. import target_objectives
from website.types import MetricType, VarType


class MyRocksParser(BaseParser):

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

    def extract_valid_variables(self, variables, catalog, default_value=None):
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
            prt_name = self.partial_name(lc_var_name)
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
        lc_variables = {self.partial_name(k.lower()): v
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
            met_info = self.metric_catalog_[self.partial_name(met_name)]
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
        return self.extract_valid_variables(
            valid_knobs, self.knob_catalog_)

    def parse_dbms_metrics(self, metrics):
        valid_metrics = self.parse_dbms_variables(metrics)
        # Extract all valid metrics
        valid_metrics, diffs = self.extract_valid_variables(
            valid_metrics, self.metric_catalog_, default_value='0')
        return valid_metrics, diffs

    def convert_dbms_metrics(self, metrics, observation_time, target_objective):
        base_metric_data = {}
        metric_data = {}
        for name, value in list(metrics.items()):
            prt_name = self.partial_name(name)

            if prt_name in self.numeric_metric_catalog_:
                metadata = self.numeric_metric_catalog_[prt_name]

                if metadata.vartype == VarType.INTEGER:
                    converted = float(self.convert_integer(value, metadata))
                elif metadata.vartype == VarType.REAL:
                    converted = self.convert_real(value, metadata)
                else:
                    raise ValueError(
                        ("Found non-numeric metric '{}' in the numeric "
                         "metric catalog: value={}, type={}").format(
                             name, value, VarType.name(metadata.vartype)))

                if metadata.metric_type == MetricType.COUNTER:
                    assert isinstance(converted, float)
                    base_metric_data[name] = converted
                    metric_data[name] = converted / observation_time
                elif metadata.metric_type == MetricType.STATISTICS:
                    assert isinstance(converted, float)
                    base_metric_data[name] = converted
                    metric_data[name] = converted
                else:
                    raise ValueError(
                        'Unknown metric type for {}: {}'.format(name, metadata.metric_type))

        target_objective_instance = target_objectives.get_instance(
            self.dbms_id, target_objective)
        metric_data[target_objective] = target_objective_instance.compute(
            base_metric_data, observation_time)

        return metric_data

    def convert_dbms_knobs(self, knobs):
        knob_data = {}
        for name, value in list(knobs.items()):
            prt_name = self.partial_name(name)
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
                            self.partial_name(k) in self.numeric_metric_catalog_])

    def filter_tunable_knobs(self, knobs):
        return OrderedDict([(k, v) for k, v in list(knobs.items()) if
                            self.partial_name(k) in self.tunable_knob_catalog_])

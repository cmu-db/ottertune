'''
Created on Dec 12, 2017

@author: dvanaken

Parser interface.
'''

import os
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict

from website.models import KnobCatalog, MetricCatalog
from website.settings import CONFIG_DIR
from website.types import BooleanType, MetricType, VarType

class BaseParser(object):

    __metaclass__ = ABCMeta

    def __init__(self, dbms_id):
        self.dbms_id_ = dbms_id
        knobs = KnobCatalog.objects.filter(dbms__pk=self.dbms_id_)
        self.knob_catalog_ = {k.name: k for k in knobs}
        self.tunable_knob_catalog_ = {k: v for k, v in \
                self.knob_catalog_.iteritems() if v.tunable is True}
        metrics = MetricCatalog.objects.filter(dbms__pk=self.dbms_id_)
        self.metric_catalog_ = {m.name: m for m in metrics}
        self.numeric_metric_catalog_ = {m: v for m, v in \
                self.metric_catalog_.iteritems() if \
                v.metric_type == MetricType.COUNTER}
        self.valid_true_val = list()
        self.valid_false_val = list()

    @abstractproperty
    def base_configuration_settings(self):
        pass

    @abstractproperty
    def knob_configuration_filename(self):
        pass

    @abstractproperty
    def transactions_counter(self):
        pass

    @abstractmethod
    def parse_version_string(self, version_string):
        pass

    def convert_bool(self, bool_value, metadata):
        if bool_value in self.valid_true_val:
            return BooleanType.TRUE
        elif bool_value in self.valid_false_val:
              return BooleanType.FALSE
        else:
            raise Exception("Invalid Boolean {}".format(bool_value))

    def convert_enum(self, enum_value, metadata):
        enumvals = metadata.enumvals.split(',')
        try:
            return enumvals.index(enum_value)
        except ValueError:
            raise Exception('Invalid enum value for variable {} ({})'.format(
                metadata.name, enum_value))

    def convert_integer(self, int_value, metadata):
        try:
            return int(int_value)
        except ValueError:
            return int(float(int_value))

    def convert_real(self, real_value, metadata):
        return float(real_value)

    def convert_string(self, string_value, metadata):
        return string_value

    def convert_timestamp(self, timestamp_value, metadata):
        return timestamp_value

    def valid_boolean_val_to_string(self):
        str_true = 'valid true values: '
        for bval in self.valid_true_val:
            str_true += str(bval) + ' '
        str_false = 'valid false values: '
        for bval in self.valid_false_val:
            str_false += str(bval) + ' '
        return str_true + '; ' + str_false

    def convert_dbms_knobs(self, knobs):
        knob_data = {}
        for name, metadata in self.tunable_knob_catalog_.iteritems():
            if metadata.tunable is False:
                continue
            if name not in knobs:
                continue
            value = knobs[name]
            conv_value = None
            if metadata.vartype == VarType.BOOL:
                if not self._check_knob_bool_val(value):
                    raise Exception('Knob boolean value not valid! '
                                    'Boolean values should be one of: {}, '
                                    'but the actual value is: {}'
                                    .format(self.valid_boolean_val_to_string(),
                                            str(value)))
                conv_value = self.convert_bool(value, metadata)
            elif metadata.vartype == VarType.ENUM:
                conv_value = self.convert_enum(value, metadata)
            elif metadata.vartype == VarType.INTEGER:
                conv_value = self.convert_integer(value, metadata)
                if not self._check_knob_num_in_range(conv_value, metadata):
                    raise Exception('Knob integer num value not in range! '
                                    'min: {}, max: {}, actual: {}'
                                    .format(metadata.minval,
                                            metadata.maxval, str(conv_value)))
            elif metadata.vartype == VarType.REAL:
                conv_value = self.convert_real(value, metadata)
                if not self._check_knob_num_in_range(conv_value, metadata):
                    raise Exception('Knob real num value not in range! '
                                    'min: {}, max: {}, actual: {}'
                                    .format(metadata.minval,
                                            metadata.maxval, str(conv_value)))
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

    def _check_knob_num_in_range(self, value, mdata):
        return value >= float(mdata.minval) and value <= float(mdata.maxval)

    def _check_knob_bool_val(self, value):
        return value in self.valid_true_val or value in self.valid_false_val

    def convert_dbms_metrics(self, metrics, observation_time):
#         if len(metrics) != len(self.numeric_metric_catalog_):
#             raise Exception('The number of metrics should be equal!')
        metric_data = {}
        for name, metadata in self.numeric_metric_catalog_.iteritems():
            value = metrics[name]
            if metadata.metric_type == MetricType.COUNTER:
                converted = self.convert_integer(value, metadata)
                metric_data[name] = float(converted) / observation_time
            else:
                raise Exception(
                    'Unknown metric type for {}: {}'.format(name, metadata.metric_type))
        if self.transactions_counter not in metric_data:
            raise Exception("Cannot compute throughput (no objective function)")
        metric_data['throughput_txn_per_sec'] = metric_data[self.transactions_counter]
        return metric_data

    @staticmethod
    def extract_valid_variables(variables, catalog, default_value=None):
        valid_variables = {}
        diff_log = []
        valid_lc_variables = {k.lower(): v for k, v in catalog.iteritems()}

        # First check that the names of all variables are valid (i.e., listed
        # in the official catalog). Invalid variables are logged as 'extras'.
        # Variable names that are valid but differ in capitalization are still
        # added to valid_variables but with the proper capitalization. They
        # are also logged as 'miscapitalized'.
        for var_name, var_value in variables.iteritems():
            lc_var_name = var_name.lower()
            if lc_var_name in valid_lc_variables:
                valid_name = valid_lc_variables[lc_var_name].name
                if var_name != valid_name:
                    diff_log.append(('miscapitalized', valid_name, var_name, var_value))
                valid_variables[valid_name] = var_value
            else:
                diff_log.append(('extra', None, var_name, var_value))

        # Next find all item names that are listed in the catalog but missing from
        # variables. Missing variables are added to valid_variables with the given
        # default_value if provided (or the item's actual default value if not) and
        # logged as 'missing'.
        lc_variables = {k.lower(): v for k, v in variables.iteritems()}
        for valid_lc_name, metadata in valid_lc_variables.iteritems():
            if valid_lc_name not in lc_variables:
                diff_log.append(('missing', metadata.name, None, None))
                valid_variables[metadata.name] = default_value if \
                    default_value is not None else metadata.default
        assert len(valid_variables) == len(catalog)
        return valid_variables, diff_log

    def parse_helper(self, scope, view_variables):
        valid_variables = {}
        for view_name, variables in view_variables.iteritems():
            for var_name, var_value in variables.iteritems():
                full_name = '{}.{}'.format(view_name, var_name)
                if full_name not in valid_variables:
                    valid_variables[full_name] = []
                valid_variables[full_name].append(var_value)
        return valid_variables

    def parse_dbms_variables(self, variables):
        valid_variables = {}
        for scope, sub_vars in variables.iteritems():
            if sub_vars is None:
                continue
            if scope == 'global':
                valid_variables.update(self.parse_helper(scope, sub_vars))
            elif scope == 'local':
                for _, viewnames in sub_vars.iteritems():
                    for viewname, objnames in viewnames.iteritems():
                        for _, view_vars in objnames.iteritems():
                            valid_variables.update(self.parse_helper(
                                scope, {viewname: view_vars}))
            else:
                raise Exception('Unsupported variable scope: {}'.format(scope))
        return valid_variables

    def parse_dbms_knobs(self, knobs):
        valid_knobs = self.parse_dbms_variables(knobs)

        for k in list(valid_knobs.keys()):
            assert len(valid_knobs[k]) == 1
            valid_knobs[k] = valid_knobs[k][0]
        # Extract all valid knobs
        return BaseParser.extract_valid_variables(
            valid_knobs, self.knob_catalog_)

    def parse_dbms_metrics(self, metrics):
        # Some DBMSs measure different types of stats (e.g., global, local)
        # at different scopes (e.g. indexes, # tables, database) so for now
        # we just combine them
        valid_metrics = self.parse_dbms_variables(metrics)

        # Extract all valid metrics
        valid_metrics, diffs = BaseParser.extract_valid_variables(
            valid_metrics, self.metric_catalog_, default_value='0')

        # Combine values
        for name, values in valid_metrics.iteritems():
            metric = self.metric_catalog_[name]
            if metric.metric_type == MetricType.INFO or len(values) == 1:
                valid_metrics[name] = values[0]
            elif metric.metric_type == MetricType.COUNTER:
                values = [int(v) for v in values if v is not None]
                if len(values) == 0:
                    valid_metrics[name] = 0
                else:
                    valid_metrics[name] = str(sum(values))
            else:
                raise Exception(
                    'Invalid metric type: {}'.format(metric.metric_type))
        return valid_metrics, diffs

    def calculate_change_in_metrics(self, metrics_start, metrics_end):
        adjusted_metrics = {}
        for met_name, start_val in metrics_start.iteritems():
            end_val = metrics_end[met_name]
            met_info = self.metric_catalog_[met_name]
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

    def create_knob_configuration(self, tuning_knobs, custom_knobs):
        config_knobs = self.base_configuration_settings
        config_knobs.update(custom_knobs)

        categories = {}
        for knob_name, knob_value in config_knobs.iteritems():
            category = self.knob_catalog_[knob_name].category
            if category not in categories:
                categories[category] = []
            categories[category].append((knob_name, knob_value))
        categories = OrderedDict(sorted(categories.iteritems()))

        config_path = os.path.join(CONFIG_DIR, self.knob_configuration_filename)
        with open(config_path, 'r') as f:
            config = f.read()

        header_fmt = ('#' + ('-' * 78) + '\n# {cat1}\n#' +
                      ('-' * 78) + '\n\n').format
        subheader_fmt = '# - {cat2} -\n\n'.format
        for category, knobs in categories.iteritems():
            parts = [p.strip() for p in category.upper().split(' / ')]
            config += header_fmt(cat1=parts[0])
            if len(parts) == 2:
                config += subheader_fmt(cat2=parts[1])
            for knob_name, knob_value in sorted(knobs):
                config += '{} = \'{}\'\n'.format(knob_name, knob_value)
            config += '\n'
        config += header_fmt(cat1='TUNING PARAMETERS')
        for knob_name, knob_value in sorted(tuning_knobs.iteritems()):
            if knob_name.startswith('global.'):
                knob_name = knob_name[len('global.'):]
            config += '{} = \'{}\'\n'.format(knob_name, knob_value)
        return config

    def get_nondefault_knob_settings(self, knobs):
        nondefault_settings = OrderedDict()
        for knob_name, metadata in self.knob_catalog_.iteritems():
            if metadata.tunable is True:
                continue
            if knob_name not in knobs:
                continue
            knob_value = knobs[knob_name]
            if knob_value != metadata.default:
                nondefault_settings[knob_name] = knob_value
        return nondefault_settings

    def format_bool(self, bool_value, metadata):
        return 'on' if bool_value == BooleanType.TRUE else 'off'

    def format_enum(self, enum_value, metadata):
        enumvals = metadata.enumvals.split(',')
        return enumvals[enum_value]

    def format_integer(self, int_value, metadata):
        return int(round(int_value))

    def format_real(self, real_value, metadata):
        return float(real_value)

    def format_string(self, string_value, metadata):
        return string_value

    def format_timestamp(self, timestamp_value, metadata):
        return timestamp_value

    def format_dbms_knobs(self, knobs):
        formatted_knobs = {}
        for knob_name, knob_value in knobs.iteritems():
            metadata = self.knob_catalog_.get(knob_name, None)
            if (metadata is None):
                raise Excepton('Unknown knob {}'.format(knob_name))
            fvalue = None
            if metadata.vartype == VarType.BOOL:
                fvalue = self.format_bool(knob_value, metadata)
            elif metadata.vartype == VarType.ENUM:
                fvalue = self.format_enum(knob_value, metadata)
            elif metadata.vartype == VarType.INTEGER:
                fvalue = self.format_integer(knob_value, metadata)
            elif metadata.vartype == VarType.REAL:
                fvalue = self.format_real(knob_value, metadata)
            elif metadata.vartype == VarType.STRING:
                fvalue = self.format_string(knob_value, metadata)
            elif metadata.vartype == VarType.TIMESTAMP:
                fvalue = self.format_timestamp(knob_value, metadata)
            else:
                raise Exception('Unknown variable type for {}: {}'.format(
                        knob_name, metadata.vartype))
            if fvalue is None:
                raise Exception('Cannot format value for {}: {}'.format(
                    knob_name, knob_value))
            formatted_knobs[knob_name] = fvalue
        return formatted_knobs

    def filter_numeric_metrics(self, metrics):
        return OrderedDict([(k, v) for k, v in metrics.iteritems() if \
                            k in self.numeric_metric_catalog_])

    def filter_tunable_knobs(self, knobs):
        return OrderedDict([(k, v) for k, v in knobs.iteritems() if \
                            k in self.tunable_knob_catalog_])

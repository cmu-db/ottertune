#
# OtterTune - parser.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from collections import OrderedDict

from website.models import KnobCatalog, KnobUnitType, MetricCatalog
from website.types import BooleanType, MetricType, VarType
from website.utils import ConversionUtil
from .. import target_objectives


# pylint: disable=no-self-use
class BaseParser:

    def __init__(self, dbms_obj):
        self.dbms_id = int(dbms_obj.pk)
        knobs = KnobCatalog.objects.filter(dbms=dbms_obj)
        self.knob_catalog_ = {k.name: k for k in knobs}
        self.tunable_knob_catalog_ = {
            k: v for k, v in self.knob_catalog_.items() if
            v.tunable is True}

        metrics = MetricCatalog.objects.filter(dbms=dbms_obj)
        self.metric_catalog_ = {m.name: m for m in metrics}
        numeric_mtypes = (MetricType.COUNTER, MetricType.STATISTICS)
        self.numeric_metric_catalog_ = {
            m: v for m, v in self.metric_catalog_.items() if
            v.metric_type in numeric_mtypes}

        self.valid_true_val = ("on", "true", "yes")
        self.valid_false_val = ("off", "false", "no")
        self.true_value = 'on'
        self.false_value = 'off'

        self.bytes_system = ConversionUtil.DEFAULT_BYTES_SYSTEM
        self.time_system = ConversionUtil.DEFAULT_TIME_SYSTEM
        self.min_bytes_unit = 'kB'
        self.min_time_unit = 'ms'

    def parse_version_string(self, version_string):
        return version_string

    def convert_bool(self, bool_value, metadata):
        if isinstance(bool_value, str):
            bool_value = bool_value.lower()

        if bool_value in self.valid_true_val:
            res = BooleanType.TRUE
        elif bool_value in self.valid_false_val:
            res = BooleanType.FALSE
        else:
            raise Exception("Invalid Boolean {}".format(bool_value))

        return res

    def convert_enum(self, enum_value, metadata):
        enumvals = metadata.enumvals.split(',')
        try:
            res = enumvals.index(enum_value)
        except ValueError:
            raise Exception('Invalid enum value for variable {} ({})'.format(
                metadata.name, enum_value))

        return res

    def convert_integer(self, int_value, metadata):
        try:
            try:
                converted = int(int_value)
            except ValueError:
                converted = int(float(int_value))

        except ValueError:
            if metadata.unit == KnobUnitType.BYTES:
                converted = ConversionUtil.get_raw_size(
                    int_value, system=self.bytes_system)
            elif metadata.unit == KnobUnitType.MILLISECONDS:
                converted = ConversionUtil.get_raw_size(
                    int_value, system=self.time_system)
            else:
                raise Exception(
                    'Unknown unit type: {}'.format(metadata.unit))
        if converted is None:
            raise Exception('Invalid integer format for {}: {}'.format(
                metadata.name, int_value))
        return converted

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
        for name, metadata in list(self.tunable_knob_catalog_.items()):
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
                raise Exception('Param value for {} cannot be null'.format(name))
            knob_data[name] = conv_value

        return knob_data

    def _check_knob_num_in_range(self, value, mdata):
        return float(mdata.minval) <= value <= float(mdata.maxval)

    def _check_knob_bool_val(self, value):
        if isinstance(str, value):
            value = value.lower()
        return value in self.valid_true_val or value in self.valid_false_val

    def convert_dbms_metrics(self, metrics, observation_time, target_objective):
        metric_data = {}
        # Same as metric_data except COUNTER metrics are not divided by the time
        base_metric_data = {}

        for name, metadata in self.numeric_metric_catalog_.items():
            value = metrics[name]

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
        lc_variables = {k.lower(): v for k, v in list(variables.items())}
        for valid_lc_name, metadata in list(valid_lc_variables.items()):
            if valid_lc_name not in lc_variables:
                diff_log.append(('missing', metadata.name, None, None))
                valid_variables[metadata.name] = default_value if \
                    default_value is not None else metadata.default
        assert len(valid_variables) == len(catalog)
        return valid_variables, diff_log

    def parse_helper(self, scope, valid_variables, view_variables):
        for view_name, variables in list(view_variables.items()):
            for var_name, var_value in list(variables.items()):
                full_name = '{}.{}'.format(view_name, var_name)
                if full_name not in valid_variables:
                    valid_variables[full_name] = []
                valid_variables[full_name].append(var_value)
        return valid_variables

    def parse_dbms_variables(self, variables):
        valid_variables = {}
        for scope, sub_vars in list(variables.items()):
            if sub_vars is None:
                continue
            if scope == 'global':
                valid_variables.update(self.parse_helper(scope, valid_variables, sub_vars))
            elif scope == 'local':
                for _, viewnames in list(sub_vars.items()):
                    for viewname, objnames in list(viewnames.items()):
                        for _, view_vars in list(objnames.items()):
                            valid_variables.update(self.parse_helper(
                                scope, valid_variables, {viewname: view_vars}))
            else:
                raise Exception('Unsupported variable scope: {}'.format(scope))
        return valid_variables

    def parse_dbms_knobs(self, knobs):
        valid_knobs = self.parse_dbms_variables(knobs)

        for k in list(valid_knobs.keys()):
            assert len(valid_knobs[k]) == 1
            valid_knobs[k] = valid_knobs[k][0]
        # Extract all valid knobs
        return self.extract_valid_variables(valid_knobs, self.knob_catalog_)

    def parse_dbms_metrics(self, metrics):
        # Some DBMSs measure different types of stats (e.g., global, local)
        # at different scopes (e.g. indexes, # tables, database) so for now
        # we just combine them
        valid_metrics = self.parse_dbms_variables(metrics)

        # Extract all valid metrics
        valid_metrics, diffs = self.extract_valid_variables(
            valid_metrics, self.metric_catalog_, default_value='0')

        # Combine values
        for name, values in list(valid_metrics.items()):
            metric = self.metric_catalog_[name]
            if metric.metric_type == MetricType.INFO or len(values) == 1:
                valid_metrics[name] = values[0]
            elif metric.metric_type == MetricType.COUNTER or \
                    metric.metric_type == MetricType.STATISTICS:
                conv_fn = int if metric.vartype == VarType.INTEGER else float
                values = [conv_fn(v) for v in values if v is not None]
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
        for met_name, start_val in list(metrics_start.items()):
            end_val = metrics_end[met_name]
            met_info = self.metric_catalog_[met_name]
            if met_info.vartype == VarType.INTEGER or \
                    met_info.vartype == VarType.REAL:
                conversion_fn = self.convert_integer if \
                    met_info.vartype == VarType.INTEGER else \
                    self.convert_real
                start_val = conversion_fn(start_val, met_info)
                end_val = conversion_fn(end_val, met_info)
                if met_info.metric_type == MetricType.COUNTER:
                    adj_val = end_val - start_val
                else:  # MetricType.STATISTICS or MetricType.INFO
                    adj_val = end_val
                assert adj_val >= 0, \
                    '{} wrong metric type: {} (start={}, end={}, diff={})'.format(
                        met_name, MetricType.name(met_info.metric_type), start_val,
                        end_val, end_val - start_val)

                adjusted_metrics[met_name] = adj_val
            else:
                # This metric is either a bool, enum, string, or timestamp
                # so take last recorded value from metrics_end
                adjusted_metrics[met_name] = end_val
        return adjusted_metrics

    def create_knob_configuration(self, tuning_knobs):
        configuration = {}
        for knob_name, knob_value in sorted(tuning_knobs.items()):
            # FIX ME: for now it only shows the global knobs, works for Postgres
            if knob_name.startswith('global.'):
                knob_name_global = knob_name[knob_name.find('.') + 1:]
                configuration[knob_name_global] = knob_value

        configuration = OrderedDict(sorted(configuration.items()))
        return configuration

    def format_bool(self, bool_value, metadata):
        return self.true_value if bool_value == BooleanType.TRUE else self.false_value

    def format_enum(self, enum_value, metadata):
        enumvals = metadata.enumvals.split(',')
        return enumvals[int(round(enum_value))]

    def format_integer(self, int_value, metadata):
        int_value = int(round(int_value))
        if metadata.unit != KnobUnitType.OTHER and int_value > 0:
            if metadata.unit == KnobUnitType.BYTES:
                int_value = ConversionUtil.get_human_readable2(
                    int_value, self.bytes_system, 'kB')
            elif metadata.unit == KnobUnitType.MILLISECONDS:
                int_value = ConversionUtil.get_human_readable2(
                    int_value, self.time_system, 'ms')
            else:
                raise Exception(
                    'Invalid unit type for {}: {}'.format(
                        metadata.name, metadata.unit))

        return int_value

    def format_real(self, real_value, metadata):
        return round(float(real_value), 3)

    def format_string(self, string_value, metadata):
        return string_value

    def format_timestamp(self, timestamp_value, metadata):
        return timestamp_value

    def format_dbms_knobs(self, knobs):
        formatted_knobs = {}
        for knob_name, knob_value in list(knobs.items()):
            metadata = self.knob_catalog_.get(knob_name, None)
            if (metadata is None):
                raise Exception('Unknown knob {}'.format(knob_name))
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
        return OrderedDict(((k, v) for k, v in list(metrics.items()) if
                            k in self.numeric_metric_catalog_))

    def filter_tunable_knobs(self, knobs):
        return OrderedDict(((k, v) for k, v in list(knobs.items()) if
                            k in self.tunable_knob_catalog_))

# pylint: enable=no-self-use

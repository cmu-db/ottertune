'''
Created on Jul 8, 2017

@author: dvanaken
'''

import json
import logging
import numpy as np
import os.path
import re
import string
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict
from random import choice
from django.utils.text import capfirst

from .models import DBMSCatalog, KnobCatalog, MetricCatalog
from .settings import CONFIG_DIR
from .types import (BooleanType, DBMSType, LabelStyleType, MetricType,
                    VarType, KnobUnitType)

LOG = logging.getLogger(__name__)


class JSONUtil(object):

    @staticmethod
    def loads(config_str):
        return json.loads(config_str,
                          encoding="UTF-8",
                          object_pairs_hook=OrderedDict)

    @staticmethod
    def dumps(config, pprint=False, sort=False):
        indent = 4 if pprint is True else None
        if sort is True:
            if isinstance(config, dict):
                config = OrderedDict(sorted(config.items()))
            else:
                config = sorted(config)

        return json.dumps(config,
                          encoding="UTF-8",
                          indent=indent)


class MediaUtil(object):
    @staticmethod
    def upload_code_generator(size=20,
                              chars=string.ascii_uppercase + string.digits):
        new_upload_code = ''.join(choice(chars) for _ in range(size))
        return new_upload_code


class DataUtil(object):

    @staticmethod
    def aggregate_data(results):
        knob_labels = JSONUtil.loads(results[0].dbms_config.data).keys()
        metric_labels = JSONUtil.loads(results[0].dbms_metrics.data).keys()
        X_matrix = np.empty((len(results), len(knob_labels)), dtype=float)
        y_matrix = np.empty((len(results), len(metric_labels)), dtype=float)
        rowlabels = np.empty(len(results), dtype=int)

        for i, result in enumerate(results):
            param_data = JSONUtil.loads(result.dbms_config.data)
            if len(param_data) != len(knob_labels):
                raise Exception(
                    ("Incorrect number of knobs "
                     "(expected={}, actual={})").format(len(knob_labels),
                                                        len(param_data)))
            metric_data = JSONUtil.loads(result.dbms_metrics.data)
            if len(metric_data) != len(metric_labels):
                raise Exception(
                    ("Incorrect number of metrics "
                     "(expected={}, actual={})").format(len(metric_labels),
                                                        len(metric_data)))
            X_matrix[i, :] = [param_data[l] for l in knob_labels]
            y_matrix[i, :] = [metric_data[l] for l in metric_labels]
            rowlabels[i] = result.pk
        return {
            'X_matrix': X_matrix,
            'y_matrix': y_matrix,
            'rowlabels': rowlabels,
            'X_columnlabels': knob_labels,
            'y_columnlabels': metric_labels,
        }

    @staticmethod
    def combine_duplicate_rows(X_matrix, y_matrix, rowlabels):
        X_unique, idxs, invs, cts = np.unique(X_matrix,
                                              return_index=True,
                                              return_inverse=True,
                                              return_counts=True,
                                              axis=0)
        num_unique = X_unique.shape[0]
        if num_unique == X_matrix.shape[0]:
            # No duplicate rows
            return X_matrix, y_matrix, rowlabels

        # Combine duplicate rows
        y_unique = np.empty((num_unique, y_matrix.shape[1]))
        rowlabels_unique = np.empty(num_unique, dtype=tuple)
        ix = np.arange(X_matrix.shape[0])
        for i, count in enumerate(cts):
            if count == 1:
                y_unique[i, :] = y_matrix[idxs[i], :]
                rowlabels_unique[i] = (rowlabels[idxs[i]],)
            else:
                dup_idxs = ix[invs == i]
                y_unique[i, :] = np.median(y_matrix[dup_idxs, :], axis=0)
                rowlabels_unique[i] = tuple(rowlabels[dup_idxs])
        return X_unique, y_unique, rowlabels_unique


class ConversionUtil(object):

    @staticmethod
    def get_raw_size(value, system):
        for factor, suffix in system:
            if value.endswith(suffix):
                if len(value) == len(suffix):
                    amount = 1
                else:
                    amount = int(value[:-len(suffix)])
                return amount * factor
        return None

    @staticmethod
    def get_human_readable(value, system):
        from hurry.filesize import size
        return size(value, system=system)


class DBMSUtilImpl(object):

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

    @abstractproperty
    def base_configuration_settings(self):
        pass

    @abstractproperty
    def configuration_filename(self):
        pass

    @abstractproperty
    def transactions_counter(self):
        pass

    @abstractmethod
    def parse_version_string(self, version_string):
        pass

    def convert_bool(self, bool_value, param_info):
        return BooleanType.TRUE if \
                bool_value.lower() == 'on' else BooleanType.FALSE

    def convert_enum(self, enum_value, param_info):
        enumvals = param_info.enumvals.split(',')
        try:
            return enumvals.index(enum_value)
        except ValueError:
            raise Exception('Invalid enum value for param {} ({})'.format(
                param_info.name, enum_value))

    def convert_integer(self, int_value, param_info):
        try:
            return int(int_value)
        except ValueError:
            return int(float(int_value))

    def convert_real(self, real_value, param_info):
        return float(real_value)

    def convert_string(self, string_value, param_info):
        raise NotImplementedError('Implement me!')

    def convert_timestamp(self, timestamp_value, param_info):
        raise NotImplementedError('Implement me!')

    def convert_dbms_params(self, params):
        param_data = {}
        for pname, pinfo in self.tunable_knob_catalog_.iteritems():
            if pinfo.tunable is False:
                continue
            if pname not in params:
                continue
            pvalue = params[pname]
            prep_value = None
            if pinfo.vartype == VarType.BOOL:
                prep_value = self.convert_bool(pvalue, pinfo)
            elif pinfo.vartype == VarType.ENUM:
                prep_value = self.convert_enum(pvalue, pinfo)
            elif pinfo.vartype == VarType.INTEGER:
                prep_value = self.convert_integer(pvalue, pinfo)
            elif pinfo.vartype == VarType.REAL:
                prep_value = self.convert_real(pvalue, pinfo)
            elif pinfo.vartype == VarType.STRING:
                prep_value = self.convert_string(pvalue, pinfo)
            elif pinfo.vartype == VarType.TIMESTAMP:
                prep_value = self.convert_timestamp(pvalue, pinfo)
            else:
                raise Exception(
                    'Unknown variable type: {}'.format(pinfo.vartype))
            if prep_value is None:
                raise Exception(
                    'Param value for {} cannot be null'.format(pname))
            param_data[pname] = prep_value
        return param_data

    def convert_dbms_metrics(self, metrics, observation_time):
#         if len(metrics) != len(self.numeric_metric_catalog_):
#             raise Exception('The number of metrics should be equal!')
        metric_data = {}
        for mname, minfo in self.numeric_metric_catalog_.iteritems():
            mvalue = metrics[mname]
            if minfo.metric_type == MetricType.COUNTER:
                converted = self.convert_integer(mvalue, minfo)
                metric_data[mname] = float(converted) / observation_time
            else:
                raise Exception(
                    'Unknown metric type: {}'.format(minfo.metric_type))
        if self.transactions_counter not in metric_data:
            raise Exception("Cannot compute throughput (no objective function)")
        metric_data['throughput_txn_per_sec'] = metric_data[self.transactions_counter]
        return metric_data

    @staticmethod
    def extract_valid_keys(idict, catalog, default=None):
        valid_dict = {}
        diffs = []
        lowercase_dict = {k.lower(): v for k, v in catalog.iteritems()}
        for k, v in idict.iteritems():
            lower_k2 = k.lower()
            if lower_k2 in lowercase_dict:
                true_k = lowercase_dict[lower_k2].name
                if k != true_k:
                    diffs.append(('miscapitalized_key', true_k, k, v))
                valid_dict[true_k] = v
            else:
                diffs.append(('extra_key', None, k, v))
        if len(idict) > len(lowercase_dict):
            assert len(diffs) > 0
        elif len(idict) < len(lowercase_dict):
            lowercase_idict = {k.lower(): v for k, v in idict.iteritems()}
            for k, v in lowercase_dict.iteritems():
                if k not in lowercase_idict:
                    # Set missing keys to a default value
                    diffs.append(('missing_key', v.name, None, None))
                    valid_dict[
                        v.name] = default if default is not None else v.default
#         assert len(valid_dict) == len(catalog)
        return valid_dict, diffs

    def parse_helper(self, config):
        valid_entries = {}
        for view_name, entries in config.iteritems():
            for mname, mvalue in entries.iteritems():
                key = '{}.{}'.format(view_name, mname)
                if key not in valid_entries:
                    valid_entries[key] = []
                valid_entries[key].append(mvalue)
        return valid_entries

    def parse_dbms_config(self, config):
        valid_knobs = {}
        for knobtype, subknobs in config.iteritems():
            if subknobs is None:
                continue
            if knobtype == 'global':
                valid_knobs.update(self.parse_helper(subknobs))
            elif knobtype == 'local':
                for scope, viewnames in subknobs.iteritems():
                    for viewname, objnames in viewnames.iteritems():
                        for objname, ssmets in objnames.iteritems():
                            valid_knobs.update(self.parse_helper({viewname: ssmets}))
            else:
                raise Exception('Unsupported knobs format: ' + knobtype)

        for k in list(valid_knobs.keys()):
            assert len(valid_knobs[k]) == 1
            valid_knobs[k] = valid_knobs[k][0]
        # Extract all valid knobs
        return DBMSUtilImpl.extract_valid_keys(valid_knobs, self.knob_catalog_, default='0')

    def parse_dbms_metrics(self, metrics):
        # Some DBMSs measure different types of stats (e.g., global, local)
        # at different scopes (e.g. indexes, # tables, database) so for now
        # we just combine them
        valid_metrics = {}
        for mettype, submetrics in metrics.iteritems():
            if submetrics is None:
                continue
            if mettype == 'global':
                valid_metrics.update(self.parse_helper(submetrics))
            elif mettype == 'local':
                for scope, viewnames in submetrics.iteritems():
                    for viewname, objnames in viewnames.iteritems():
                        for objname, ssmets in objnames.iteritems():
                            valid_metrics.update(self.parse_helper({viewname: ssmets}))
            else:
                raise Exception('Unsupported metrics format: ' + mettype)

        # Extract all valid metrics
        valid_metrics, diffs = DBMSUtilImpl.extract_valid_keys(
            valid_metrics, self.metric_catalog_, default='0')

        # Combine values
        for mname, mvalues in valid_metrics.iteritems():
            metric = self.metric_catalog_[mname]
            mvalues = valid_metrics[mname]
            if metric.metric_type == MetricType.INFO or len(mvalues) == 1:
                valid_metrics[mname] = mvalues[0]
            elif metric.metric_type == MetricType.COUNTER:
                mvalues = [int(v) for v in mvalues if v is not None]
                if len(mvalues) == 0:
                    valid_metrics[mname] = 0
                else:
                    valid_metrics[mname] = str(sum(mvalues))
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

    def create_configuration(self, tuning_params, custom_params):
        config_params = self.base_configuration_settings
        config_params.update(custom_params)

        categories = {}
        for pname, pvalue in config_params.iteritems():
            category = self.knob_catalog_[pname].category
            if category not in categories:
                categories[category] = []
            categories[category].append((pname, pvalue))
        categories = OrderedDict(sorted(categories.iteritems()))

        config_path = os.path.join(CONFIG_DIR, self.configuration_filename)
        with open(config_path, 'r') as f:
            config = f.read()

        header_fmt = ('#' + ('-' * 78) + '\n# {cat1}\n#' +
                      ('-' * 78) + '\n\n').format
        subheader_fmt = '# - {cat2} -\n\n'.format
        for category, params in categories.iteritems():
            parts = [p.strip() for p in category.upper().split(' / ')]
            config += header_fmt(cat1=parts[0])
            if len(parts) == 2:
                config += subheader_fmt(cat2=parts[1])
            for pname, pval in sorted(params):
                config += '{} = \'{}\'\n'.format(pname, pval)
            config += '\n'
        config += header_fmt(cat1='TUNING PARAMETERS')
        for pname, pval in sorted(tuning_params.iteritems()):
            if pname.startswith('global.'):
                pname = pname[len('global.'):]
            config += '{} = \'{}\'\n'.format(pname, pval)
        return config

    def get_nondefault_settings(self, config):
        nondefault_settings = OrderedDict()
        for pname, pinfo in self.knob_catalog_.iteritems():
            if pinfo.tunable is True:
                continue
            if pname not in config:
                continue
            pvalue = config[pname]
            if pvalue != pinfo.default:
                nondefault_settings[pname] = pvalue
        return nondefault_settings

    def format_bool(self, bool_value, param_info):
        return 'on' if bool_value == BooleanType.TRUE else 'off'

    def format_enum(self, enum_value, param_info):
        enumvals = param_info.enumvals.split(',')
        return enumvals[enum_value]

    def format_integer(self, int_value, param_info):
        return int(round(int_value))

    def format_real(self, real_value, param_info):
        return float(real_value)

    def format_string(self, string_value, param_info):
        raise NotImplementedError('Implement me!')

    def format_timestamp(self, timestamp_value, param_info):
        raise NotImplementedError('Implement me!')

    def format_dbms_params(self, params):
        formatted_params = {}
        for pname, pvalue in params.iteritems():
            pinfo = self.knob_catalog_[pname]
            prep_value = None
            if pinfo.vartype == VarType.BOOL:
                prep_value = self.format_bool(pvalue, pinfo)
            elif pinfo.vartype == VarType.ENUM:
                prep_value = self.format_enum(pvalue, pinfo)
            elif pinfo.vartype == VarType.INTEGER:
                prep_value = self.format_integer(pvalue, pinfo)
            elif pinfo.vartype == VarType.REAL:
                prep_value = self.format_real(pvalue, pinfo)
            elif pinfo.vartype == VarType.STRING:
                prep_value = self.format_string(pvalue, pinfo)
            elif pinfo.vartype == VarType.TIMESTAMP:
                prep_value = self.format_timestamp(pvalue, pinfo)
            else:
                raise Exception(
                    'Unknown variable type: {}'.format(pinfo.vartype))
            if prep_value is None:
                raise Exception(
                    'Cannot format value for {}'.format(pname))
            formatted_params[pname] = prep_value
        return formatted_params

    def filter_numeric_metrics(self, metrics, normalize=False):
        return OrderedDict([(k, v) for k, v in metrics.iteritems() if \
                            k in self.numeric_metric_catalog_])

    def filter_tunable_params(self, params):
        return OrderedDict([(k, v) for k, v in params.iteritems() if \
                            k in self.tunable_knob_catalog_])


class PostgresUtilImpl(DBMSUtilImpl):

    POSTGRES_BYTES_SYSTEM = [
        (1024 ** 5, 'PB'),
        (1024 ** 4, 'TB'),
        (1024 ** 3, 'GB'),
        (1024 ** 2, 'MB'),
        (1024 ** 1, 'kB'),
        (1024 ** 0, 'B'),
    ]

    POSTGRES_TIME_SYSTEM = [
        (1000 * 60 * 60 * 24, 'd'),
        (1000 * 60 * 60, 'h'),
        (1000 * 60, 'min'),
        (1, 'ms'),
        (1000, 's'),
    ]

    POSTGRES_BASE_PARAMS = {
        'global.data_directory': None,
        'global.hba_file': None,
        'global.ident_file': None,
        'global.external_pid_file': None,
        'global.listen_addresses': None,
        'global.port': None,
        'global.max_connections': None,
        'global.unix_socket_directories': None,
        'global.log_line_prefix': '%t [%p-%l] %q%u@%d ',
        'global.track_counts': 'on',
        'global.track_io_timing': 'on',
        'global.autovacuum': 'on',
        'global.default_text_search_config': 'pg_catalog.english',
    }

    @property
    def base_configuration_settings(self):
        return dict(self.POSTGRES_BASE_PARAMS)

    @property
    def configuration_filename(self):
        return 'postgresql.conf'

    @property
    def transactions_counter(self):
        return 'pg_stat_database.xact_commit'

    def convert_integer(self, int_value, param_info):
        converted = None
        try:
            converted = super(PostgresUtilImpl, self).convert_integer(
                int_value, param_info)
        except ValueError:
            if param_info.unit == KnobUnitType.BYTES:
                converted = ConversionUtil.get_raw_size(
                    int_value, system=self.POSTGRES_BYTES_SYSTEM)
            elif param_info.unit == KnobUnitType.MILLISECONDS:
                converted = ConversionUtil.get_raw_size(
                    int_value, system=self.POSTGRES_TIME_SYSTEM)
            else:
                raise Exception(
                    'Unknown unit type: {}'.format(param_info.unit))
        if converted is None:
            raise Exception('Invalid integer format for param {} ({})'.format(
                param_info.name, int_value))
        return converted

    def format_integer(self, int_value, param_info):
        if param_info.unit != KnobUnitType.OTHER and int_value > 0:
            if param_info.unit == KnobUnitType.BYTES:
                int_value = ConversionUtil.get_human_readable(
                    int_value, PostgresUtilImpl.POSTGRES_BYTES_SYSTEM)
            elif param_info.unit == KnobUnitType.MILLISECONDS:
                int_value = ConversionUtil.get_human_readable(
                    int_value, PostgresUtilImpl.POSTGRES_TIME_SYSTEM)
            else:
                raise Exception(
                    'Invalid knob unit type: {}'.format(param_info.unit))
        else:
            int_value = super(PostgresUtilImpl, self).format_integer(int_value, param_info)
        return int_value

    def parse_version_string(self, version_string):
        dbms_version = version_string.split(',')[0]
        return re.search("\d+\.\d+(?=\.\d+)", dbms_version).group(0)


class Postgres96UtilImpl(PostgresUtilImpl):

    def __init__(self):
        dbms = DBMSCatalog.objects.get(
            type=DBMSType.POSTGRES, version='9.6')
        super(Postgres96UtilImpl, self).__init__(dbms.pk)


class DBMSUtil(object):

    __DBMS_UTILS_IMPLS = None

    @staticmethod
    def __utils(dbms_id=None):
        if DBMSUtil.__DBMS_UTILS_IMPLS is None:
            DBMSUtil.__DBMS_UTILS_IMPLS = {
                DBMSCatalog.objects.get(
                    type=DBMSType.POSTGRES, version='9.6').pk: Postgres96UtilImpl()
            } 
        try:
            if dbms_id is None:
                return DBMSUtil.__DBMS_UTILS_IMPLS
            else:
                return DBMSUtil.__DBMS_UTILS_IMPLS[dbms_id]
        except KeyError:
            raise NotImplementedError(
                'Implement me! ({})'.format(dbms_id))

    @staticmethod
    def parse_version_string(dbms_type, version_string):
        for k, v in DBMSUtil.__utils(dbms_type).iteritems():
            dbms = DBMSCatalog.objects.get(pk=k)
            if dbms.type == dbms_type:
                try:
                    return v.parse_version_string(version_string)
                except:
                    pass
        return None

    @staticmethod
    def convert_dbms_params(dbms_id, params):
        return DBMSUtil.__utils(dbms_id).convert_dbms_params(
                params)

    @staticmethod
    def convert_dbms_metrics(dbms_id, numeric_metrics, observation_time):
        return DBMSUtil.__utils(dbms_id).convert_dbms_metrics(
                numeric_metrics, observation_time)

    @staticmethod
    def parse_dbms_config(dbms_id, config):
        return DBMSUtil.__utils(dbms_id).parse_dbms_config(config)

    @staticmethod
    def parse_dbms_metrics(dbms_id, metrics):
        return DBMSUtil.__utils(dbms_id).parse_dbms_metrics(metrics)

    @staticmethod
    def get_nondefault_settings(dbms_id, config):
        return DBMSUtil.__utils(dbms_id).get_nondefault_settings(
            config)

    @staticmethod
    def create_configuration(dbms_id, tuning_params, custom_params):
        return DBMSUtil.__utils(dbms_id).create_configuration(
            tuning_params, custom_params)

    @staticmethod
    def format_dbms_params(dbms_id, params):
        return DBMSUtil.__utils(dbms_id).format_dbms_params(params)

    @staticmethod
    def get_configuration_filename(dbms_id):
        return DBMSUtil.__utils(dbms_id).configuration_filename

    @staticmethod
    def filter_numeric_metrics(dbms_id, metrics, normalize=False):
        return DBMSUtil.__utils(dbms_id).filter_numeric_metrics(
            metrics, normalize)

    @staticmethod
    def filter_tunable_params(dbms_id, params):
        return DBMSUtil.__utils(dbms_id).filter_tunable_params(params)

    @staticmethod
    def calculate_change_in_metrics(dbms_id, metrics_start, metrics_end):
        return DBMSUtil.__utils(dbms_id).calculate_change_in_metrics(
            metrics_start, metrics_end)


class LabelUtil(object):

    @staticmethod
    def style_labels(label_map, style=LabelStyleType.DEFAULT_STYLE):
        style_labels = {}
        for name, verbose_name in label_map.iteritems():
            if style == LabelStyleType.TITLE:
                label = verbose_name.title()
            elif style == LabelStyleType.CAPFIRST:
                label = capfirst(verbose_name)
            elif style == LabelStyleType.LOWER:
                label = verbose_name.lower()
            else:
                raise Exception('Invalid style: {}'.format(style))
            if style != LabelStyleType.LOWER and 'dbms' in label.lower():
                label = label.replace('dbms', 'DBMS')
                label = label.replace('Dbms', 'DBMS')
            style_labels[name] = label
        return style_labels

#
# OtterTune - utils.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import datetime
import json
import logging
import math
import os
import string
import tarfile
import time
from collections import OrderedDict
from io import BytesIO
from random import choice

import numpy as np
from django.utils.text import capfirst
from django_db_logger.models import StatusLog
from djcelery.models import TaskMeta

from .models import DBMSCatalog, KnobCatalog, Result, Session, SessionKnob
from .settings import constants
from .types import LabelStyleType, VarType

LOG = logging.getLogger(__name__)


class JSONUtil(object):

    @staticmethod
    def loads(config_str):
        return json.loads(config_str,
                          encoding="UTF-8",
                          object_pairs_hook=OrderedDict)

    @staticmethod
    def dumps(config, pprint=False, sort=False, encoder='custom'):
        json_args = dict(indent=4 if pprint is True else None,
                         ensure_ascii=False)

        if encoder == 'custom':
            json_args.update(default=JSONUtil.custom_converter)

        if sort is True:
            if isinstance(config, dict):
                config = OrderedDict(sorted(config.items()))
            else:
                config = sorted(config)

        return json.dumps(config, **json_args)

    @staticmethod
    def custom_converter(o):
        if isinstance(o, datetime.datetime):
            return str(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()


class MediaUtil(object):

    @staticmethod
    def upload_code_generator(size=20,
                              chars=string.ascii_uppercase + string.digits):
        new_upload_code = ''.join(choice(chars) for _ in range(size))
        return new_upload_code


class TaskUtil(object):

    @staticmethod
    def get_tasks(tasks):
        if not tasks:
            return []
        task_ids = tasks.split(',')
        res = []
        for task_id in task_ids:
            task = TaskMeta.objects.filter(task_id=task_id)
            if len(task) == 0:
                continue  # Task Not Finished
            res.append(task[0])
        return res

    @staticmethod
    def get_task_status(tasks):
        if len(tasks) == 0:
            return None, 0
        overall_status = 'SUCCESS'
        num_completed = 0
        for task in tasks:
            status = task.status
            if status == "SUCCESS":
                num_completed += 1
            elif status in ['FAILURE', 'REVOKED', 'RETRY']:
                overall_status = status
                break
            else:
                assert status in ['PENDING', 'RECEIVED', 'STARTED']
                overall_status = status
        return overall_status, num_completed


class DataUtil(object):

    @staticmethod
    def get_knob_bounds(knob_labels, session):
        minvals = []
        maxvals = []
        for _, knob in enumerate(knob_labels):
            knob_object = KnobCatalog.objects.get(dbms=session.dbms, name=knob, tunable=True)
            knob_session_object = SessionKnob.objects.filter(knob=knob_object, session=session,
                                                             tunable=True)
            if knob_session_object.exists():
                minval = float(knob_session_object[0].minval)
                maxval = float(knob_session_object[0].maxval)
            else:
                minval = float(knob_object.minval)
                maxval = float(knob_object.maxval)
            minvals.append(minval)
            maxvals.append(maxval)
        return np.array(minvals), np.array(maxvals)

    @staticmethod
    def aggregate_data(results):
        knob_labels = list(JSONUtil.loads(results[0].knob_data.data).keys())
        metric_labels = list(JSONUtil.loads(results[0].metric_data.data).keys())
        X_matrix = np.empty((len(results), len(knob_labels)), dtype=float)
        y_matrix = np.empty((len(results), len(metric_labels)), dtype=float)
        rowlabels = np.empty(len(results), dtype=int)

        for i, result in enumerate(results):
            param_data = JSONUtil.loads(result.knob_data.data)
            if len(param_data) != len(knob_labels):
                raise Exception(
                    ("Incorrect number of knobs "
                     "(expected={}, actual={})").format(len(knob_labels),
                                                        len(param_data)))
            metric_data = JSONUtil.loads(result.metric_data.data)
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
            'rowlabels': rowlabels.tolist(),
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

            # For consistency, tuple the rowlabels
            rowlabels = np.array([tuple([x]) for x in rowlabels])  # pylint: disable=bad-builtin,deprecated-lambda
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

    @staticmethod
    def dummy_encoder_helper(featured_knobs, dbms):
        n_values = []
        cat_knob_indices = []
        cat_knob_names = []
        noncat_knob_names = []
        binary_knob_indices = []
        dbms_info = DBMSCatalog.objects.filter(pk=dbms.pk)

        if len(dbms_info) == 0:
            raise Exception("DBMSCatalog cannot find dbms {}".format(dbms.full_name()))
        full_dbms_name = dbms_info[0]

        for i, knob_name in enumerate(featured_knobs):
            # knob can be uniquely identified by (dbms, knob_name)
            knobs = KnobCatalog.objects.filter(name=knob_name,
                                               dbms=dbms)
            if len(knobs) == 0:
                raise Exception(
                    "KnobCatalog cannot find knob of name {} in {}".format(
                        knob_name, full_dbms_name))
            knob = knobs[0]
            # check if knob is ENUM
            if knob.vartype == VarType.ENUM:
                # enumvals is a comma delimited list
                enumvals = knob.enumvals.split(",")
                if len(enumvals) > 2:
                    # more than 2 values requires dummy encoding
                    n_values.append(len(enumvals))
                    cat_knob_indices.append(i)
                    cat_knob_names.append(knob_name)
                else:
                    # knob is binary
                    noncat_knob_names.append(knob_name)
                    binary_knob_indices.append(i)
            else:
                if knob.vartype == VarType.BOOL:
                    binary_knob_indices.append(i)
                noncat_knob_names.append(knob_name)

        n_values = np.array(n_values)
        cat_knob_indices = np.array(cat_knob_indices)
        categorical_info = {'n_values': n_values,
                            'categorical_features': cat_knob_indices,
                            'cat_columnlabels': cat_knob_names,
                            'noncat_columnlabels': noncat_knob_names,
                            'binary_vars': binary_knob_indices}
        return categorical_info


class ConversionUtil(object):

    DEFAULT_BYTES_SYSTEM = (
        (1024 ** 5, 'PB'),
        (1024 ** 4, 'TB'),
        (1024 ** 3, 'GB'),
        (1024 ** 2, 'MB'),
        (1024 ** 1, 'kB'),
        (1024 ** 0, 'B'),
    )

    DEFAULT_TIME_SYSTEM = (
        (1000 * 60 * 60 * 24, 'd'),
        (1000 * 60 * 60, 'h'),
        (1000 * 60, 'min'),
        (1000, 's'),
        (1, 'ms'),
    )

    @staticmethod
    def get_raw_size(value, system):
        for factor, suffix in system:
            if value.endswith(suffix):
                if len(value) == len(suffix):
                    amount = 1
                else:
                    try:
                        amount = int(value[:-len(suffix)])
                    except ValueError:
                        continue
                return amount * factor
        return None

    @staticmethod
    def get_human_readable(value, system):
        from hurry.filesize import size
        return size(value, system=system)

    @staticmethod
    def get_human_readable2(value, system, min_suffix):
        # Converts the value to larger units only if there is no loss of resolution.
        # pylint: disable=line-too-long
        # From https://github.com/le0pard/pgtune/blob/master/webpack/components/configurationView/index.jsx#L74
        # pylint: enable=line-too-long
        min_factor = None
        unit = None
        mod_system = []
        for i, (factor, suffix) in enumerate(system):
            if suffix == min_suffix:
                if value < factor:
                    assert i + 1 < len(system), \
                        ('i + 1 >= len(system) (i + 1: {}, len(system): {}, value: {}, '
                         'min_suffix: {})').format(i + 1, len(system), value, min_suffix)
                    min_suffix = system[i + 1][1]
                    LOG.warning('The value is smaller than the min factor: %s < %s%s. '
                                'Setting min_suffix=%s...', value, factor, suffix, min_suffix)
                else:
                    min_factor = factor
                    unit = min_suffix
                    value = math.floor(float(value) / min_factor)
                    break

            mod_system.append((factor, suffix))

        if min_factor is None:
            raise ValueError('Invalid min suffix for system: suffix={}, system={}'.format(
                min_suffix, system))

        LOG.info('min_suffix: %s, min_factor: %s, unit: %s, value: %s\nMOD_SYS:\n%s\n\n',
                 min_suffix, min_factor, unit, value, mod_system)
        for factor, suffix in mod_system:
            adj_factor = factor / min_factor
            if value % adj_factor == 0:
                value = math.floor(float(value) / adj_factor)
                unit = suffix
                break

        return '{}{}'.format(int(value), unit)


class LabelUtil(object):

    @staticmethod
    def style_labels(label_map, style=LabelStyleType.DEFAULT_STYLE):
        style_labels = {}
        for name, verbose_name in list(label_map.items()):
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
            style_labels[name] = str(label)
        return style_labels


def dump_debug_info(session, pretty_print=False):
    files = {}

    # Session
    session_values = Session.objects.filter(pk=session.pk).values()[0]
    session_values['dbms'] = session.dbms.full_name
    session_values['hardware'] = session.hardware.name

    # Session knobs
    knob_instances = SessionKnob.objects.filter(
        session=session, tunable=True).select_related('knob')
    knob_values = list(knob_instances.values())
    for knob, knob_dict in zip(knob_instances, knob_values):
        assert knob.pk == knob_dict['id']
        knob_dict['knob'] = knob.name
    session_values['knobs'] = knob_values

    # Save binary field types to separate files
    binary_fields = [
        'ddpg_actor_model',
        'ddpg_critic_model',
        'ddpg_reply_memory',
        'dnn_model',
    ]
    for bf in binary_fields:
        if session_values[bf]:
            filename = os.path.join('binaries', '{}.pickle'.format(bf))
            content = session_values[bf]
            session_values[bf] = filename
            files[filename] = content

    files['session.json'] = session_values

    # Results from session
    result_instances = Result.objects.filter(session=session).select_related(
        'knob_data', 'metric_data').order_by('creation_time')
    results = []

    for result, result_dict in zip(result_instances, result_instances.values()):
        assert result.pk == result_dict['id']
        result_dict = OrderedDict(result_dict)
        next_config = result.next_configuration or '{}'
        result_dict['next_configuration'] = JSONUtil.loads(next_config)

        tasks = {}
        task_ids = result.task_ids
        task_ids = task_ids.split(',') if task_ids else []
        for task_id in task_ids:
            task = TaskMeta.objects.filter(task_id=task_id).values()
            task = task[0] if task else None
            tasks[task_id] = task
        result_dict['tasks'] = tasks

        knob_data = result.knob_data.data or '{}'
        metric_data = result.metric_data.data or '{}'
        result_dict['knob_data'] = JSONUtil.loads(knob_data)
        result_dict['metric_data'] = JSONUtil.loads(metric_data)
        results.append(result_dict)

    files['results.json'] = results

    # Log messages written to the database using django-db-logger
    logs = StatusLog.objects.filter(create_datetime__gte=session.creation_time)
    logger_names = logs.order_by().values_list('logger_name', flat=True).distinct()

    # Write log files at app scope (e.g., django, website, celery)
    logger_names = set([l.split('.', 1)[0] for l in logger_names])

    for logger_name in logger_names:
        log_values = list(logs.filter(logger_name__startswith=logger_name).order_by(
            'create_datetime').values())
        for lv in log_values:
            lv['level'] = logging.getLevelName(lv['level'])
        files['logs/{}.log'.format(logger_name)] = log_values

    # Save settings
    constants_dict = OrderedDict()
    for name, value in sorted(constants.__dict__.items()):
        if not name.startswith('_') and name == name.upper():
            constants_dict[name] = value
    files['constants.json'] = constants_dict

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = 'debug_{}'.format(timestamp)

    mtime = time.time()
    tarstream = BytesIO()
    with tarfile.open(mode='w:gz', fileobj=tarstream) as tar:
        for filename, content in files.items():  # pylint: disable=not-an-iterable
            if isinstance(content, (dict, list)):
                content = JSONUtil.dumps(content, pprint=pretty_print)
            if isinstance(content, str):
                content = content.encode('utf-8')
            assert isinstance(content, bytes), (filename, type(content))
            bio = BytesIO(content)
            path = os.path.join(root, filename)
            tarinfo = tarfile.TarInfo(name=path)
            tarinfo.size = len(bio.getvalue())
            tarinfo.mtime = mtime
            tar.addfile(tarinfo, bio)

    tarstream.seek(0)
    return tarstream, root

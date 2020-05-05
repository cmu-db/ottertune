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
from subprocess import Popen, PIPE

import celery
import numpy as np
from django.contrib.auth.models import User
from django.core.management import call_command
from django.db.models import Case, When
from django.utils.text import capfirst
from django_db_logger.models import StatusLog
from djcelery.models import TaskMeta

from .models import DBMSCatalog, MetricCatalog, KnobCatalog, Result, Session, SessionKnob
from .settings import common
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
    def get_task_ids_from_tuple(task_tuple):
        task_res = celery.result.result_from_tuple(task_tuple)
        task_ids = []
        task = task_res
        while task is not None:
            task_ids.insert(0, task)
            task = task.parent
        return task_ids

    @staticmethod
    def get_tasks(task_ids):
        task_ids = task_ids or []
        if isinstance(task_ids, str):
            task_ids = task_ids.split(',')
        preserved = Case(*[
            When(task_id=task_id, then=pos) for pos, task_id in enumerate(task_ids)])
        return TaskMeta.objects.filter(task_id__in=task_ids).order_by(preserved)

    @staticmethod
    def get_task_status(tasks, num_tasks):
        overall_status = 'UNAVAILABLE'
        num_completed = 0
        for task in tasks:
            status = task.status
            if status == "SUCCESS":
                num_completed += 1
            elif status in ('FAILURE', 'REVOKED', 'RETRY'):
                overall_status = status
                break
            else:
                if status not in ('PENDING', 'RECEIVED', 'STARTED'):
                    LOG.warning("Task %s: invalid task status: '%s' (task_id=%s)",
                                task.id, status, task.task_id)
                overall_status = status

        if num_tasks > 0 and num_tasks == num_completed:
            overall_status = 'SUCCESS'

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
            if knob_object.vartype is VarType.ENUM:
                enumvals = knob_object.enumvals.split(',')
                minval = 0
                maxval = len(enumvals) - 1
            elif knob_object.vartype is VarType.BOOL:
                minval = 0
                maxval = 1
            elif knob_session_object.exists():
                minval = float(knob_session_object[0].minval)
                maxval = float(knob_session_object[0].maxval)
            else:
                minval = float(knob_object.minval)
                maxval = float(knob_object.maxval)
            minvals.append(minval)
            maxvals.append(maxval)
        return np.array(minvals), np.array(maxvals)

    @staticmethod
    def aggregate_data(results, ignore=None):
        if ignore is None:
            ignore = ['range_test']
        knob_labels = sorted(JSONUtil.loads(results[0].knob_data.data).keys())
        metric_labels = sorted(JSONUtil.loads(results[0].metric_data.data).keys())
        X_matrix = []
        y_matrix = []
        rowlabels = []

        for result in results:
            if any(symbol in result.metric_data.name for symbol in ignore):
                continue
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

            X_matrix.append([param_data[l] for l in knob_labels])
            y_matrix.append([metric_data[l] for l in metric_labels])
            rowlabels.append(result.pk)
        return {
            'X_matrix': np.array(X_matrix, dtype=np.float64),
            'y_matrix': np.array(y_matrix, dtype=np.float64),
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
    def clean_metric_data(metric_matrix, metric_labels, useful_views):
        # Make metric_labels identical to useful_labels (if not None)
        if useful_views is None:
            return metric_matrix, metric_labels

        useful_labels = []
        for label in metric_labels:
            for view in useful_views:
                if view in label:
                    useful_labels.append(label)
                    break

        missing_columns = sorted(set(useful_labels) - set(metric_labels))
        unused_columns = set(metric_labels) - set(useful_labels)
        LOG.debug("clean_metric_data: added %d metrics and removed %d metric.",
                  len(missing_columns), len(unused_columns))
        default_val = 0
        useful_labels_size = len(useful_labels)
        matrix = np.ones((len(metric_matrix), useful_labels_size)) * default_val
        metric_labels_dict = {n: i for i, n in enumerate(metric_labels)}
        # column labels in matrix has the same order as ones in useful_labels
        # missing values are filled with default_val
        for i, metric_name in enumerate(useful_labels):
            if metric_name in metric_labels_dict:
                index = metric_labels_dict[metric_name]
                matrix[:, i] = metric_matrix[:, index]
        LOG.debug("clean_metric_data: final ~ matrix: %s, labels: %s", matrix.shape,
                  useful_labels_size)
        return matrix, useful_labels

    @staticmethod
    def clean_knob_data(knob_matrix, knob_labels, sessions):
        # Filter and amend knob_matrix and knob_labels according to the tunable knobs in the session
        knob_matrix = np.array(knob_matrix)
        session_knobs = []
        knob_cat = []
        for session in sessions:
            knobs_for_this_session = SessionKnob.objects.get_knobs_for_session(session)
            for knob in knobs_for_this_session:
                if knob['name'] not in knob_cat:
                    session_knobs.append(knob)
            knob_cat = [k['name'] for k in session_knobs]

        if len(knob_cat) == 0 or knob_cat == knob_labels:
            # Nothing to do!
            return knob_matrix, knob_labels

        LOG.info("session_knobs: %s, knob_labels: %s, missing: %s, extra: %s", len(knob_cat),
                 len(knob_labels), len(set(knob_cat) - set(knob_labels)),
                 len(set(knob_labels) - set(knob_cat)))

        nrows = knob_matrix.shape[0]  # pylint: disable=unsubscriptable-object
        new_labels = []
        new_columns = []

        for knob in session_knobs:
            knob_name = knob['name']
            if knob_name not in knob_labels:
                # Add missing column initialized to knob's default value
                default_val = knob['default']
                try:
                    if knob['vartype'] == VarType.ENUM:
                        default_val = knob['enumvals'].split(',').index(default_val)
                    elif knob['vartype'] == VarType.BOOL:
                        default_val = str(default_val).lower() in ("on", "true", "yes", "0")
                    else:
                        default_val = float(default_val)
                except ValueError:
                    LOG.warning("Error parsing knob '%s' default value: %s. Setting default to 0.",
                                knob_name, default_val, exc_info=True)
                    default_val = 0
                new_col = np.ones((nrows, 1), dtype=float) * default_val
                new_lab = knob_name
            else:
                index = knob_labels.index(knob_name)
                new_col = knob_matrix[:, index].reshape(-1, 1)
                new_lab = knob_labels[index]

            new_labels.append(new_lab)
            new_columns.append(new_col)

        new_matrix = np.hstack(new_columns).reshape(nrows, -1)
        LOG.debug("Cleaned matrix: %s, knobs (%s): %s", new_matrix.shape,
                  len(new_labels), new_labels)

        assert new_labels == knob_cat, \
            "Expected knobs: {}\nActual knobs:  {}\n".format(
                knob_cat, new_labels)
        assert new_matrix.shape == (nrows, len(knob_cat)), \
            "Expected shape: {}, Actual shape:  {}".format(
                (nrows, len(knob_cat)), new_matrix.shape)

        return new_matrix, new_labels

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
                    if i + 1 >= len(system):
                        LOG.warning("Error converting value '%s': min_suffix='%s' at index='%s' "
                                    "is already the smallest suffix.", value, min_suffix, i)
                        return value

                    min_suffix = system[i + 1][1]
                    LOG.warning('The value is smaller than the min factor: %s < %s (1%s). '
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


def create_user(username, password, email=None, superuser=False):
    user = User.objects.filter(username=username).first()
    if user:
        created = False
    else:
        if superuser:
            email = email or '{}@noemail.com'.format(username)
            _create_user = User.objects.create_superuser
        else:
            _create_user = User.objects.create_user

        user = _create_user(username=username, password=password, email=email)
        created = True

    return user, created


def delete_user(username):
    user = User.objects.filter(username=username).first()
    if user:
        delete_info = user.delete()
        deleted = True
    else:
        delete_info = None
        deleted = False

    return delete_info, deleted


def model_to_dict2(m, exclude=None):
    exclude = exclude or []
    d = {}
    for f in m._meta.fields:  # pylint: disable=protected-access
        fname = f.name
        if fname not in exclude:
            d[fname] = getattr(m, fname, None)
    return d


def check_and_run_celery():
    celery_status = os.popen('python3 manage.py celery inspect ping').read()
    if 'pong' in celery_status:
        return 'celery is running'

    rabbitmq_url = common.BROKER_URL.split('@')[-1]
    hostname = rabbitmq_url.split(':')[0]
    port = rabbitmq_url.split(':')[1].split('/')[0]
    rabbitmq_status = os.popen('telnet {} {}'.format(hostname, port)).read()
    LOG.info(rabbitmq_status)

    retries = 0
    while retries < 5:
        LOG.warning('Celery is not running.')
        retries += 1
        call_command('stopcelery')
        os.popen('python3 manage.py startcelery &')
        time.sleep(30 * retries)
        celery_status = os.popen('python3 manage.py celery inspect ping').read()
        if 'pong' in celery_status:
            LOG.info('Successfully start celery.')
            return 'celery stopped but is restarted successfully'
    LOG.warning('Cannot restart celery.')
    return 'celery stopped and cannot be restarted'


def git_hash():
    sha = ''
    if os.path.exists('/app/.git_commit'):
        with open('/app/.git_commit', 'r') as f:
            lines = f.read().strip().split('\n')
        for line in lines:
            if line.startswith('base='):
                sha = line.strip().split('=', 1)[1]
                break
    else:
        try:
            p = Popen("git log -1 --format=format:%H", shell=True, stdout=PIPE, stderr=PIPE)
            sha = p.communicate()[0].decode('utf-8')
        except OSError as e:
            LOG.warning("Failed to get git commit hash.\n\n%s\n\n", e, exc_info=True)

    return sha

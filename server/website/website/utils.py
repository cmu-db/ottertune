#
# OtterTune - utils.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Jul 8, 2017

@author: dvanaken
'''

import json
import logging
import string
from collections import OrderedDict
from random import choice

import numpy as np
from django.utils.text import capfirst
from djcelery.models import TaskMeta

from .types import LabelStyleType, VarType
from .models import KnobCatalog, DBMSCatalog

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
                          ensure_ascii=False,
                          indent=indent)


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

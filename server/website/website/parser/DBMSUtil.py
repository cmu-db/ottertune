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

from website.models import DBMSCatalog, KnobCatalog, MetricCatalog
from website.settings import CONFIG_DIR
from website.types import (BooleanType, DBMSType, LabelStyleType, MetricType,
                    VarType, KnobUnitType)

from website.utils import ConversionUtil
from .Postgres import Postgres96UtilImpl

LOG = logging.getLogger(__name__)

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

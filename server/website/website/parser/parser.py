#
# OtterTune - parser.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Dec 12, 2017

@author: dvanaken
'''

from .myrocks import MyRocks56Parser
from .postgres import Postgres96Parser
from website.models import DBMSCatalog
from website.types import DBMSType


class Parser(object):

    __DBMS_UTILS_IMPLS = None

    @staticmethod
    def __utils(dbms_id=None):
        if Parser.__DBMS_UTILS_IMPLS is None:
            Parser.__DBMS_UTILS_IMPLS = {
                DBMSCatalog.objects.get(
                    type=DBMSType.POSTGRES, version='9.6').pk: Postgres96Parser(),
                DBMSCatalog.objects.get(
                    type=DBMSType.MYROCKS, version='5.6').pk: MyRocks56Parser()
            }
        try:
            if dbms_id is None:
                return Parser.__DBMS_UTILS_IMPLS
            else:
                return Parser.__DBMS_UTILS_IMPLS[dbms_id]
        except KeyError:
            raise NotImplementedError(
                'Implement me! ({})'.format(dbms_id))

    @staticmethod
    def parse_version_string(dbms_type, version_string):
        for k, v in Parser.__utils(dbms_type).iteritems():
            dbms = DBMSCatalog.objects.get(pk=k)
            if dbms.type == dbms_type:
                try:
                    return v.parse_version_string(version_string)
                except AttributeError:
                    pass
        return None

    @staticmethod
    def convert_dbms_knobs(dbms_id, knobs):
        return Parser.__utils(dbms_id).convert_dbms_knobs(knobs)

    @staticmethod
    def convert_dbms_metrics(dbms_id, numeric_metrics, observation_time):
        return Parser.__utils(dbms_id).convert_dbms_metrics(
            numeric_metrics, observation_time)

    @staticmethod
    def parse_dbms_knobs(dbms_id, knobs):
        return Parser.__utils(dbms_id).parse_dbms_knobs(knobs)

    @staticmethod
    def parse_dbms_metrics(dbms_id, metrics):
        return Parser.__utils(dbms_id).parse_dbms_metrics(metrics)

    @staticmethod
    def get_nondefault_knob_settings(dbms_id, knobs):
        return Parser.__utils(dbms_id).get_nondefault_knob_settings(knobs)

    @staticmethod
    def create_knob_configuration(dbms_id, tuning_knobs, custom_knobs):
        return Parser.__utils(dbms_id).create_knob_configuration(
            tuning_knobs, custom_knobs)

    @staticmethod
    def format_dbms_knobs(dbms_id, knobs):
        return Parser.__utils(dbms_id).format_dbms_knobs(knobs)

    @staticmethod
    def get_knob_configuration_filename(dbms_id):
        return Parser.__utils(dbms_id).knob_configuration_filename

    @staticmethod
    def filter_numeric_metrics(dbms_id, metrics):
        return Parser.__utils(dbms_id).filter_numeric_metrics(metrics)

    @staticmethod
    def filter_tunable_knobs(dbms_id, knobs):
        return Parser.__utils(dbms_id).filter_tunable_knobs(knobs)

    @staticmethod
    def calculate_change_in_metrics(dbms_id, metrics_start, metrics_end):
        return Parser.__utils(dbms_id).calculate_change_in_metrics(
            metrics_start, metrics_end)

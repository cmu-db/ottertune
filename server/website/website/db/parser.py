#
# OtterTune - parser.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from website.models import DBMSCatalog
from website.types import DBMSType

from .myrocks.parser import MyRocksParser
from .postgres.parser import PostgresParser
from .oracle.parser import OracleParser

_DBMS_PARSERS = {}


def _get(dbms_id):
    dbms_id = int(dbms_id)
    db_parser = _DBMS_PARSERS.get(dbms_id, None)
    if db_parser is None:
        obj = DBMSCatalog.objects.get(id=dbms_id)
        if obj.type == DBMSType.POSTGRES:
            clz = PostgresParser
        elif obj.type == DBMSType.MYROCKS:
            clz = MyRocksParser
        elif obj.type == DBMSType.ORACLE:
            clz = OracleParser
        else:
            raise NotImplementedError('Implement me! {}'.format(obj))

        db_parser = clz(obj)
        _DBMS_PARSERS[dbms_id] = db_parser

    return db_parser


def parse_version_string(dbms_type, version_string):
    dbmss = DBMSCatalog.objects.filter(type=dbms_type)
    parsed_version = None
    for instance in dbmss:
        db_parser = _get(instance.pk)
        try:
            parsed_version = db_parser.parse_version_string(version_string)
        except AttributeError:
            pass
        if parsed_version is not None:
            break
    return parsed_version


def convert_dbms_knobs(dbms_id, knobs):
    return _get(dbms_id).convert_dbms_knobs(knobs)


def convert_dbms_metrics(dbms_id, numeric_metrics, observation_time, target_objective):
    return _get(dbms_id).convert_dbms_metrics(
        numeric_metrics, observation_time, target_objective)


def parse_dbms_knobs(dbms_id, knobs):
    return _get(dbms_id).parse_dbms_knobs(knobs)


def parse_dbms_metrics(dbms_id, metrics):
    return _get(dbms_id).parse_dbms_metrics(metrics)


def create_knob_configuration(dbms_id, tuning_knobs):
    return _get(dbms_id).create_knob_configuration(tuning_knobs)


def format_dbms_knobs(dbms_id, knobs):
    return _get(dbms_id).format_dbms_knobs(knobs)


def filter_numeric_metrics(dbms_id, metrics):
    return _get(dbms_id).filter_numeric_metrics(metrics)


def filter_tunable_knobs(dbms_id, knobs):
    return _get(dbms_id).filter_tunable_knobs(knobs)


def calculate_change_in_metrics(dbms_id, metrics_start, metrics_end):
    return _get(dbms_id).calculate_change_in_metrics(
        metrics_start, metrics_end)

#
# OtterTune - types.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Jul 9, 2017

@author: dvanaken
'''

from collections import OrderedDict


class BaseType(object):
    TYPE_NAMES = {}

    @classmethod
    def choices(cls):
        return list(cls.TYPE_NAMES.items())

    @classmethod
    def name(cls, ctype):
        return cls.TYPE_NAMES[ctype]

    @classmethod
    def type(cls, name):
        return [k for k, v in list(cls.TYPE_NAMES.items()) if
                v.lower() == name.lower()][0]


class DBMSType(BaseType):
    MYSQL = 1
    POSTGRES = 2
    DB2 = 3
    ORACLE = 4
    SQLSERVER = 5
    SQLITE = 6
    HSTORE = 7
    VECTOR = 8
    MYROCKS = 9

    TYPE_NAMES = {
        MYSQL: 'MySQL',
        POSTGRES: 'Postgres',
        DB2: 'Db2',
        ORACLE: 'Oracle',
        SQLITE: 'SQLite',
        HSTORE: 'HStore',
        VECTOR: 'Vector',
        SQLSERVER: 'SQL Server',
        MYROCKS: 'MyRocks',
    }


class MetricType(BaseType):
    COUNTER = 1
    INFO = 2
    STATISTICS = 3

    TYPE_NAMES = {
        COUNTER: 'COUNTER',
        INFO: 'INFO',
        STATISTICS: 'STATISTICS',
    }


class VarType(BaseType):
    STRING = 1
    INTEGER = 2
    REAL = 3
    BOOL = 4
    ENUM = 5
    TIMESTAMP = 6

    TYPE_NAMES = {
        STRING: 'STRING',
        INTEGER: 'INTEGER',
        REAL: 'REAL',
        BOOL: 'BOOL',
        ENUM: 'ENUM',
        TIMESTAMP: 'TIMESTAMP',
    }


class WorkloadStatusType(BaseType):
    MODIFIED = 1
    PROCESSING = 2
    PROCESSED = 3

    TYPE_NAMES = {
        MODIFIED: 'MODIFIED',
        PROCESSING: 'PROCESSING',
        PROCESSED: 'PROCESSED'
    }


class TaskType(BaseType):
    PREPROCESS = 1
    RUN_WM = 2
    RUN_GPR = 3

    # Should be in order of execution!!
    TYPE_NAMES = OrderedDict([
        (PREPROCESS, "Preprocess"),
        (RUN_WM, "Workload Mapping"),
        (RUN_GPR, "GPR"),
    ])


class BooleanType(BaseType):
    TRUE = int(True)
    FALSE = int(False)

    TYPE_NAMES = {
        TRUE: str(True),
        FALSE: str(False),
    }


class KnobUnitType(BaseType):
    BYTES = 1
    MILLISECONDS = 2
    OTHER = 3

    TYPE_NAMES = {
        BYTES: 'bytes',
        MILLISECONDS: 'milliseconds',
        OTHER: 'other',
    }


class KnobResourceType(BaseType):
    MEMORY = 1
    CPU = 2
    STORAGE = 3
    OTHER = 4

    TYPE_NAMES = {
        MEMORY: 'Memory',
        CPU: 'CPU',
        STORAGE: 'Storage',
        OTHER: 'Other',
    }


class PipelineTaskType(BaseType):
    PRUNED_METRICS = 1
    RANKED_KNOBS = 2
    KNOB_DATA = 3
    METRIC_DATA = 4

    TYPE_NAMES = {
        PRUNED_METRICS: "Pruned Metrics",
        RANKED_KNOBS: "Ranked Knobs",
        KNOB_DATA: "Knob Data",
        METRIC_DATA: "Metric Data",
    }


class LabelStyleType(BaseType):
    TITLE = 0
    CAPFIRST = 1
    LOWER = 2

    DEFAULT_STYLE = TITLE

    TYPE_NAMES = {
        TITLE: 'title',
        CAPFIRST: 'capfirst',
        LOWER: 'lower'
    }

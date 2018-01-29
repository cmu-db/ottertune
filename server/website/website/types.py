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
        return list(cls.TYPE_NAMES.iteritems())

    @classmethod
    def name(cls, ctype):
        return cls.TYPE_NAMES[ctype]

    @classmethod
    def type(cls, name):
        return [k for k, v in cls.TYPE_NAMES.iteritems() if
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

    TYPE_NAMES = {
        COUNTER: 'COUNTER',
        INFO: 'INFO',
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


class HardwareType(BaseType):
    GENERIC = 1
    EC2_T2NANO = 2
    EC2_T2MICRO = 3
    EC2_T2SMALL = 4
    EC2_T2MEDIUM = 5
    EC2_T2LARGE = 6
    EC2_T2XLARGE = 7
    EC2_T22XLARGE = 8
    EC2_M4LARGE = 9
    EC2_M4XLARGE = 10
    EC2_M42XLARGE = 11
    EC2_M44XLARGE = 12
    EC2_M410XLARGE = 13
    EC2_M416XLARGE = 14
    EC2_M3MEDIUM = 15
    EC2_M3LARGE = 16
    EC2_M3XLARGE = 17
    EC2_M32XLARGE = 18
    EC2_C4LARGE = 19
    EC2_C4XLARGE = 20
    EC2_C42XLARGE = 21
    EC2_C44XLARGE = 22
    EC2_C48XLARGE = 23
    EC2_C3LARGE = 24
    EC2_C3XLARGE = 25
    EC2_C32XLARGE = 26
    EC2_C34XLARGE = 27
    EC2_C38XLARGE = 28
    EC2_P2XLARGE = 29
    EC2_P28XLARGE = 30
    EC2_P216XLARGE = 31
    EC2_G22XLARGE = 32
    EC2_G28XLARGE = 33
    EC2_X116LARGE = 34
    EC2_X132XLARGE = 35
    EC2_R4LARGE = 36
    EC2_R4XLARGE = 37
    EC2_R42XLARGE = 38
    EC2_R44XLARGE = 39
    EC2_R48XLARGE = 40
    EC2_R416XLARGE = 41
    EC2_R3LARGE = 42
    EC2_R3XLARGE = 43
    EC2_R32XLARGE = 44
    EC2_R34XLARGE = 45
    EC2_R38XLARGE = 46
    EC2_I3LARGE = 47
    EC2_I3XLARGE = 48
    EC2_I32XLARGE = 49
    EC2_I34XLARGE = 50
    EC2_I38XLARGE = 51
    EC2_I316LARGE = 52
    EC2_D2XLARGE = 53
    EC2_D22XLARGE = 54
    EC2_D24XLARGE = 55
    EC2_D28XLARGE = 56
    EC2_F12XLARGE = 57
    EC2_F116XLARGE = 58

    TYPE_NAMES = {
        GENERIC: 'generic',
        EC2_T2NANO: 't2.nano',
        EC2_T2MICRO: 't2.micro',
        EC2_T2SMALL: 't2.small',
        EC2_T2MEDIUM: 't2.medium',
        EC2_T2LARGE: 't2.large',
        EC2_T2XLARGE: 't2.xlarge',
        EC2_T22XLARGE: 't2.2xlarge',
        EC2_M4LARGE: 'm4.large',
        EC2_M4XLARGE: 'm4.xlarge',
        EC2_M42XLARGE: 'm4.2xlarge',
        EC2_M44XLARGE: 'm4.4xlarge',
        EC2_M410XLARGE: 'm4.10xlarge',
        EC2_M416XLARGE: 'm4.16xlarge',
        EC2_M3MEDIUM: 'm3.medium',
        EC2_M3LARGE: 'm3.large',
        EC2_M3XLARGE: 'm3.xlarge',
        EC2_M32XLARGE: 'm3.2xlarge',
        EC2_C4LARGE: 'c4.large',
        EC2_C4XLARGE: 'c4.xlarge',
        EC2_C42XLARGE: 'c4.2xlarge',
        EC2_C44XLARGE: 'c4.4xlarge',
        EC2_C48XLARGE: 'c4.8xlarge',
        EC2_C3LARGE: 'c3.large',
        EC2_C3XLARGE: 'c3.xlarge',
        EC2_C32XLARGE: 'c3.2xlarge',
        EC2_C34XLARGE: 'c3.4xlarge',
        EC2_C38XLARGE: 'c3.8xlarge',
        EC2_P2XLARGE: 'p2.xlarge',
        EC2_P28XLARGE: 'p2.8xlarge',
        EC2_P216XLARGE: 'p2.16xlarge',
        EC2_G22XLARGE: 'g2.2xlarge',
        EC2_G28XLARGE: 'g2.8xlarge',
        EC2_X116LARGE: 'x1.16large',
        EC2_X132XLARGE: 'x1.32xlarge',
        EC2_R4LARGE: 'r4.large',
        EC2_R4XLARGE: 'r4.xlarge',
        EC2_R42XLARGE: 'r4.2xlarge',
        EC2_R44XLARGE: 'r4.4xlarge',
        EC2_R48XLARGE: 'r4.8xlarge',
        EC2_R416XLARGE: 'r4.16xlarge',
        EC2_R3LARGE: 'r3.large',
        EC2_R3XLARGE: 'r3.xlarge',
        EC2_R32XLARGE: 'r3.2xlarge',
        EC2_R34XLARGE: 'r3.4xlarge',
        EC2_R38XLARGE: 'r3.8xlarge',
        EC2_I3LARGE: 'i3.large',
        EC2_I3XLARGE: 'i3.xlarge',
        EC2_I32XLARGE: 'i3.2xlarge',
        EC2_I34XLARGE: 'i3.4xlarge',
        EC2_I38XLARGE: 'i3.8xlarge',
        EC2_I316LARGE: 'i3.16large',
        EC2_D2XLARGE: 'd2.xlarge',
        EC2_D22XLARGE: 'd2.2xlarge',
        EC2_D24XLARGE: 'd2.4xlarge',
        EC2_D28XLARGE: 'd2.8xlarge',
        EC2_F12XLARGE: 'f1.2xlarge',
        EC2_F116XLARGE: 'f1.16xlarge',
    }

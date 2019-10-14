#
# OtterTune - set_default_knobs.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import logging

from .models import KnobCatalog, SessionKnob
from .types import DBMSType, KnobResourceType, VarType

LOG = logging.getLogger(__name__)

# Default tunable knobs by DBMS. If a DBMS is not listed here, the set of
# tunable knobs in the KnobCatalog will be used instead.
DEFAULT_TUNABLE_KNOBS = {
    DBMSType.POSTGRES: {
        "global.checkpoint_completion_target",
        "global.default_statistics_target",
        "global.effective_cache_size",
        "global.maintenance_work_mem",
        "global.max_wal_size",
        "global.max_worker_processes",
        "global.shared_buffers",
        "global.temp_buffers",
        "global.wal_buffers",
        "global.work_mem",
    }
}

# Bytes in a GB
GB = 1024 ** 3

# Default minval when set to None
MINVAL = 0

# Default maxval when set to None
MAXVAL = 192 * GB

# Percentage of total CPUs to use for maxval
CPU_PERCENT = 2.0

# Percentage of total memory to use for maxval
MEMORY_PERCENT = 0.8

# Percentage of total storage to use for maxval
STORAGE_PERCENT = 0.8

# The maximum connections to the database
SESSION_NUM = 50.0


def set_default_knobs(session):
    dbtype = session.dbms.type
    default_tunable_knobs = DEFAULT_TUNABLE_KNOBS.get(dbtype)

    if not default_tunable_knobs:
        default_tunable_knobs = set(KnobCatalog.objects.filter(
            dbms=session.dbms, tunable=True).values_list('name', flat=True))

    for knob in KnobCatalog.objects.filter(dbms=session.dbms):
        tunable = knob.name in default_tunable_knobs
        minval = knob.minval

        if knob.vartype in (VarType.INTEGER, VarType.REAL):
            vtype = int if knob.vartype == VarType.INTEGER else float

            minval = vtype(minval) if minval is not None else MINVAL
            knob_maxval = vtype(knob.maxval) if knob.maxval is not None else MAXVAL

            if knob.resource == KnobResourceType.CPU:
                maxval = session.hardware.cpu * CPU_PERCENT
            elif knob.resource == KnobResourceType.MEMORY:
                minval = session.hardware.memory * minval
                maxval = session.hardware.memory * GB * MEMORY_PERCENT
            elif knob.resource == KnobResourceType.STORAGE:
                minval = session.hardware.storage * minval
                maxval = session.hardware.storage * GB * STORAGE_PERCENT
            else:
                maxval = knob_maxval

            # Special cases
            if dbtype == DBMSType.POSTGRES:
                if knob.name in ('global.work_mem', 'global.temp_buffers'):
                    maxval /= SESSION_NUM

            if maxval > knob_maxval:
                maxval = knob_maxval

            if maxval < minval:
                LOG.warning(("Invalid range for session knob '%s': maxval <= minval "
                             "(minval: %s, maxval: %s). Setting maxval to the vendor setting: %s."),
                            knob.name, minval, maxval, knob_maxval)
                maxval = knob_maxval

            maxval = vtype(maxval)

        else:
            assert knob.resource == KnobResourceType.OTHER
            maxval = knob.maxval

        SessionKnob.objects.create(session=session,
                                   knob=knob,
                                   minval=minval,
                                   maxval=maxval,
                                   tunable=tunable)

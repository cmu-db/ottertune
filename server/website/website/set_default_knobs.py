#
# OtterTune - set_default_knobs.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import logging

from .models import KnobCatalog, SessionKnob
from .types import DBMSType, KnobResourceType, VarType

LOG = logging.getLogger(__name__)

GB = 1024 ** 3

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

            minval = vtype(minval)
            knob_maxval = vtype(knob.maxval)

            if knob.resource == KnobResourceType.CPU:
                maxval = session.hardware.cpu * 2
            elif knob.resource == KnobResourceType.MEMORY:
                maxval = session.hardware.memory * GB
            elif knob.resource == KnobResourceType.STORAGE:
                maxval = session.hardware.storage * GB
            else:
                maxval = knob_maxval

            # Special cases
            if dbtype == DBMSType.POSTGRES:
                if knob.name == 'global.work_mem':
                    maxval /= 50.0

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

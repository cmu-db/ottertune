#
# OtterTune - set_default_knobs.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from .models import KnobCatalog, SessionKnob
from .types import DBMSType


def turn_knobs_off(session, knob_names):
    for knob_name in knob_names:
        knob = KnobCatalog.objects.filter(dbms=session.dbms, name=knob_name).first()
        SessionKnob.objects.create(session=session,
                                   knob=knob,
                                   minval=knob.minval,
                                   maxval=knob.maxval,
                                   tunable=False)


def set_knob_tuning_range(session, knob_name, minval, maxval):
    knob = KnobCatalog.objects.filter(dbms=session.dbms, name=knob_name).first()
    SessionKnob.objects.create(session=session,
                               knob=knob,
                               minval=minval,
                               maxval=maxval,
                               tunable=True)


def set_default_knobs(session):
    if session.dbms.type == DBMSType.POSTGRES and session.dbms.version == '9.6':
        turn_knobs_off(session, ["global.backend_flush_after", "global.bgwriter_delay",
                                 "global.bgwriter_flush_after", "global.bgwriter_lru_multiplier",
                                 "global.checkpoint_flush_after", "global.commit_delay",
                                 "global.commit_siblings", "global.deadlock_timeout",
                                 "global.effective_io_concurrency", "global.from_collapse_limit",
                                 "global.join_collapse_limit", "global.maintenance_work_mem",
                                 "global.max_worker_processes",
                                 "global.min_parallel_relation_size", "global.min_wal_size",
                                 "global.seq_page_cost", "global.wal_buffers",
                                 "global.wal_sync_method", "global.wal_writer_delay",
                                 "global.wal_writer_flush_after"])

        set_knob_tuning_range(session, "global.checkpoint_completion_target", 0.1, 0.9)
        set_knob_tuning_range(session, "global.checkpoint_timeout", 60000, 1800000)
        set_knob_tuning_range(session, "global.default_statistics_target", 100, 2048)
        set_knob_tuning_range(session, "global.effective_cache_size", 4294967296, 17179869184)
        set_knob_tuning_range(session, "global.max_parallel_workers_per_gather", 0, 8)
        set_knob_tuning_range(session, "global.max_wal_size", 268435456, 17179869184)
        set_knob_tuning_range(session, "global.random_page_cost", 1, 10)
        set_knob_tuning_range(session, "global.shared_buffers", 134217728, 12884901888)
        set_knob_tuning_range(session, "global.temp_buffers", 8388608, 1073741824)
        set_knob_tuning_range(session, "global.work_mem", 4194304, 1073741824)

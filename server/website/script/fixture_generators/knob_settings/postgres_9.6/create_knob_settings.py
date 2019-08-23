#
# OtterTune - create_knob_settings.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import csv
import json
import shutil
from collections import OrderedDict

PG_SYSTEM = [
    (1024 ** 5, 'PB'),
    (1024 ** 4, 'TB'),
    (1024 ** 3, 'GB'),
    (1024 ** 2, 'MB'),
    (1024 ** 1, 'kB'),
    (1024 ** 0, 'B'),
]

PG_TIME = [
    (1000 * 1 * 60, 'min'),
    (1000 ** 0, 'ms'),
    (1000 ** 1, 's'),
]

# def create_tuning_config(t_minval=None, t_maxval=None, t_minval_type=None, t_maxval_type=None,
#                          t_resource_type=None, t_weight_samples=False,
#                          t_step=None, t_enumvals=None,
#                          t_powers_of_2=False, t_additional_values=[], t_dependent=False,
#                          t_notes=''):
#     cfg = {}
#     cfg['t_minval'] = t_minval
#     cfg['t_minval_type'] = t_minval_type
#     cfg['t_maxval'] = t_maxval
#     cfg['t_maxval_type'] = t_maxval_type
#     cfg['t_resource_type'] = t_resource_type
#     cfg['t_step'] = t_step
#     cfg['t_enumvals'] = t_enumvals
#     cfg['t_powers_of_2'] = t_powers_of_2
#     cfg['t_additional_values'] = t_additional_values
#     cfg['t_dependent'] = t_dependent
#     cfg['t_weight_samples'] = t_weight_samples
#
#     return cfg


STRING = 1
INTEGER = 2
REAL = 3
BOOL = 4
ENUM = 5
TIMESTAMP = 6

TYPE_NAMES = {
    'string': STRING,
    'integer': INTEGER,
    'real': REAL,
    'bool': BOOL,
    'enum': ENUM,
    'timestamp': TIMESTAMP
}

UNIT_BYTES = 1
UNIT_MS = 2
UNIT_OTHER = 3


def convert(size, system=None):
    if system is None:
        system = PG_SYSTEM
    for multiplier, suffix in system:
        if size.endswith(suffix):
            if len(size) == len(suffix):
                amount = 1
            else:
                amount = int(size[:-len(suffix)])
            return amount * multiplier
    return None


PARAMS = OrderedDict()
PARAM_PREFIX = 'global'

with open("settings.csv", "r") as f:
    READER = csv.READER(f, delimiter=',')
    HEADER = None
    for i, row in enumerate(READER):
        if i == 0:
            HEADER = list(row)
        else:
            param = {}
            param['name'] = row[HEADER.index('name')]
            param['vartype'] = TYPE_NAMES[row[HEADER.index('vartype')]]
            param['category'] = row[HEADER.index('category')]
            param['enumvals'] = row[HEADER.index('enumvals')]

            param['context'] = row[HEADER.index('context')]
            param['unit'] = None
            param['tunable'] = None
            param['scope'] = 'global'
            param['summary'] = row[HEADER.index('short_desc')]
            param['description'] = row[HEADER.index('extra_desc')]

            default = row[HEADER.index('boot_val')]
            minval = row[HEADER.index('min_val')]
            maxval = row[HEADER.index('max_val')]
            if param['vartype'] == INTEGER:
                default = int(default)
                minval = int(minval)
                maxval = int(maxval)
            elif param['vartype'] == REAL:
                default = float(default)  # pylint: disable=redefined-variable-type
                minval = float(minval)    # pylint: disable=redefined-variable-type
                maxval = float(maxval)    # pylint: disable=redefined-variable-type
            else:
                assert minval == ''
                assert maxval == ''
                minval = None
                maxval = None

            param['minval'] = minval
            param['maxval'] = maxval
            param['default'] = default

            if param['enumvals'] != '':
                enumvals = param['enumvals'][1:-1].split(',')
                for i, enumval in enumerate(enumvals):
                    if enumval.startswith('\"') and enumval.endswith('\"'):
                        enumvals[i] = enumval[1:-1]
                param['enumvals'] = ','.join(enumvals)
            else:
                param['enumvals'] = None

            pg_unit = row[HEADER.index('unit')]
            if pg_unit != '':
                factor = convert(pg_unit)
                if factor is None:
                    factor = convert(pg_unit, system=PG_TIME)
                    assert factor is not None
                    param['unit'] = UNIT_MS
                else:
                    param['unit'] = UNIT_BYTES

                if param['default'] > 0:
                    param['default'] = param['default'] * factor
                if param['minval'] > 0:
                    param['minval'] = param['minval'] * factor
                if param['maxval'] > 0:
                    param['maxval'] = param['maxval'] * factor
            else:
                param['unit'] = UNIT_OTHER

            # Internal params are read-only
            if param['context'] == 'internal':
                param['tunable'] = 'no'

            # All string param types are not tunable in 9.6
            if param['vartype'] == STRING:
                param['tunable'] = 'no'

            # We do not tune autovacuum (yet)
            if param['name'].startswith('autovacuum'):
                param['tunable'] = 'no'

            # No need to tune debug params
            if param['name'].startswith('debug'):
                param['tunable'] = 'no'

            # Don't want to disable query tuning options
            if param['name'].startswith('enable'):
                param['tunable'] = 'no'

            # These options control a special-case query optimizer
            if param['name'].startswith('geqo'):
                param['tunable'] = 'no'

            # Do not tune logging settings
            if param['name'].startswith('log'):
                param['tunable'] = 'no'

            # Do not tune SSL settings
            if param['name'].startswith('ssl'):
                param['tunable'] = 'no'

            # Do not tune syslog settings
            if param['name'].startswith('syslog'):
                param['tunable'] = 'no'

            # Do not tune TPC settings
            if param['name'].startswith('tcp'):
                param['tunable'] = 'no'

            if param['name'].startswith('trace'):
                param['tunable'] = 'no'

            if param['name'].startswith('track'):
                param['tunable'] = 'no'

            # We do not tune autovacuum (yet)
            if param['name'].startswith('vacuum'):
                param['tunable'] = 'no'

            # Do not tune replication settings
            if param['category'].startswith('Replication'):
                param['tunable'] = 'no'

            PARAMS[param['name']] = param

# We only want to tune some settings
PARAMS['allow_system_table_mods']['tunable'] = 'no'
PARAMS['archive_mode']['tunable'] = 'no'
PARAMS['archive_timeout']['tunable'] = 'no'
PARAMS['array_nulls']['tunable'] = 'no'
PARAMS['authentication_timeout']['tunable'] = 'no'
PARAMS['backend_flush_after']['tunable'] = 'yes'
PARAMS['backslash_quote']['tunable'] = 'no'
PARAMS['bgwriter_delay']['tunable'] = 'yes'
PARAMS['bgwriter_flush_after']['tunable'] = 'yes'
PARAMS['bgwriter_lru_maxpages']['tunable'] = 'yes'
PARAMS['bgwriter_lru_multiplier']['tunable'] = 'yes'
PARAMS['bonjour']['tunable'] = 'no'
PARAMS['bonjour_name']['tunable'] = 'no'
PARAMS['bytea_output']['tunable'] = 'no'
PARAMS['check_function_bodies']['tunable'] = 'no'
PARAMS['checkpoint_completion_target']['tunable'] = 'yes'
PARAMS['checkpoint_flush_after']['tunable'] = 'yes'
PARAMS['checkpoint_timeout']['tunable'] = 'yes'
PARAMS['checkpoint_warning']['tunable'] = 'no'
PARAMS['client_min_messages']['tunable'] = 'no'
PARAMS['commit_delay']['tunable'] = 'yes'
PARAMS['commit_siblings']['tunable'] = 'yes'
PARAMS['constraint_exclusion']['tunable'] = 'no'
PARAMS['cpu_index_tuple_cost']['tunable'] = 'maybe'
PARAMS['cpu_operator_cost']['tunable'] = 'maybe'
PARAMS['cpu_tuple_cost']['tunable'] = 'maybe'
PARAMS['cursor_tuple_fraction']['tunable'] = 'maybe'
PARAMS['db_user_namespace']['tunable'] = 'no'
PARAMS['deadlock_timeout']['tunable'] = 'yes'
PARAMS['default_statistics_target']['tunable'] = 'yes'
PARAMS['default_transaction_deferrable']['tunable'] = 'no'
PARAMS['default_transaction_isolation']['tunable'] = 'no'
PARAMS['default_transaction_read_only']['tunable'] = 'no'
PARAMS['default_with_oids']['tunable'] = 'no'
PARAMS['dynamic_shared_memory_type']['tunable'] = 'no'
PARAMS['effective_cache_size']['tunable'] = 'yes'
PARAMS['effective_io_concurrency']['tunable'] = 'yes'
PARAMS['escape_string_warning']['tunable'] = 'no'
PARAMS['exit_on_error']['tunable'] = 'no'
PARAMS['extra_float_digits']['tunable'] = 'no'
PARAMS['force_parallel_mode']['tunable'] = 'no'
PARAMS['from_collapse_limit']['tunable'] = 'yes'
PARAMS['fsync']['tunable'] = 'no'  # dangerous
PARAMS['full_page_writes']['tunable'] = 'no'  # dangerous
PARAMS['gin_fuzzy_search_limit']['tunable'] = 'no'
PARAMS['gin_pending_list_limit']['tunable'] = 'no'
PARAMS['huge_pages']['tunable'] = 'no'
PARAMS['idle_in_transaction_session_timeout']['tunable'] = 'no'
PARAMS['ignore_checksum_failure']['tunable'] = 'no'
PARAMS['ignore_system_indexes']['tunable'] = 'no'
PARAMS['IntervalStyle']['tunable'] = 'no'
PARAMS['join_collapse_limit']['tunable'] = 'yes'
PARAMS['krb_caseins_users']['tunable'] = 'no'
PARAMS['lo_compat_privileges']['tunable'] = 'no'
PARAMS['lock_timeout']['tunable'] = 'no'  # Tuning is not recommended in Postgres 9.6 manual
PARAMS['maintenance_work_mem']['tunable'] = 'yes'
PARAMS['max_connections']['tunable'] = 'no'  # This is set based on # of client connections
PARAMS['max_files_per_process']['tunable'] = 'no'  # Should only be increased if OS complains
PARAMS['max_locks_per_transaction']['tunable'] = 'no'
PARAMS['max_parallel_workers_per_gather']['tunable'] = 'yes'  # Must be < max_worker_processes
PARAMS['max_pred_locks_per_transaction']['tunable'] = 'no'
PARAMS['max_prepared_transactions']['tunable'] = 'no'
PARAMS['max_replication_slots']['tunable'] = 'no'
PARAMS['max_stack_depth']['tunable'] = 'no'
PARAMS['max_wal_senders']['tunable'] = 'no'
PARAMS['max_wal_size']['tunable'] = 'yes'
PARAMS['max_worker_processes']['tunable'] = 'yes'
PARAMS['min_parallel_relation_size']['tunable'] = 'yes'
PARAMS['min_wal_size']['tunable'] = 'yes'
PARAMS['old_snapshot_threshold']['tunable'] = 'no'
PARAMS['operator_precedence_warning']['tunable'] = 'no'
PARAMS['parallel_setup_cost']['tunable'] = 'maybe'
PARAMS['parallel_tuple_cost']['tunable'] = 'maybe'
PARAMS['password_encryption']['tunable'] = 'no'
PARAMS['port']['tunable'] = 'no'
PARAMS['post_auth_delay']['tunable'] = 'no'
PARAMS['pre_auth_delay']['tunable'] = 'no'
PARAMS['quote_all_identifiers']['tunable'] = 'no'
PARAMS['random_page_cost']['tunable'] = 'yes'
PARAMS['replacement_sort_tuples']['tunable'] = 'no'
PARAMS['restart_after_crash']['tunable'] = 'no'
PARAMS['row_security']['tunable'] = 'no'
PARAMS['seq_page_cost']['tunable'] = 'yes'
PARAMS['session_replication_role']['tunable'] = 'no'
PARAMS['shared_buffers']['tunable'] = 'yes'
PARAMS['sql_inheritance']['tunable'] = 'no'
PARAMS['standard_conforming_strings']['tunable'] = 'no'
PARAMS['statement_timeout']['tunable'] = 'no'
PARAMS['superuser_reserved_connections']['tunable'] = 'no'
PARAMS['synchronize_seqscans']['tunable'] = 'no'
PARAMS['synchronous_commit']['tunable'] = 'no'  # dangerous
PARAMS['temp_buffers']['tunable'] = 'yes'
PARAMS['temp_file_limit']['tunable'] = 'no'
PARAMS['transaction_deferrable']['tunable'] = 'no'
PARAMS['transaction_isolation']['tunable'] = 'no'
PARAMS['transaction_read_only']['tunable'] = 'no'
PARAMS['transform_null_equals']['tunable'] = 'no'
PARAMS['unix_socket_permissions']['tunable'] = 'no'
PARAMS['update_process_title']['tunable'] = 'no'
PARAMS['wal_buffers']['tunable'] = 'yes'
PARAMS['wal_compression']['tunable'] = 'no'
PARAMS['wal_keep_segments']['tunable'] = 'no'
PARAMS['wal_level']['tunable'] = 'no'
PARAMS['wal_log_hints']['tunable'] = 'no'
PARAMS['wal_sync_method']['tunable'] = 'yes'
PARAMS['wal_writer_delay']['tunable'] = 'yes'
PARAMS['wal_writer_flush_after']['tunable'] = 'yes'
PARAMS['work_mem']['tunable'] = 'yes'
PARAMS['xmlbinary']['tunable'] = 'no'
PARAMS['xmloption']['tunable'] = 'no'
PARAMS['zero_damaged_pages']['tunable'] = 'no'


with open('tunable_params.txt', 'w') as f:
    for opt in ['yes', 'maybe', 'no', '']:
        f.write(opt.upper() + '\n')
        f.write('---------------------------------------------------\n')
        for p, pdict in list(PARAMS.items()):
            if pdict['tunable'] == opt:
                f.write('{}\t{}\t{}\n'.format(p, pdict['vartype'], pdict['unit']))
        f.write('\n')

# MAX_MEM = 36  # 64GB or 2^36
#
# # backend_flush_after - range between 0 & 2MB
# # max = 2^21, eff_min = 2^13 (8kB), step either 0.5 or 1
# # other_values = [0]
# # powers_of_2 = true
# PARAMS['backend_flush_after']['tuning_config'] = create_tuning_config(
#     t_minval=13, t_maxval=21, t_step=0.5, t_additional_values=[0],
#     t_powers_of_2=True, t_weight_samples=True)
#
# # bgwriter_delay
# # true minval = 10, maxval = 500, step = 10
# PARAMS['bgwriter_delay']['tuning_config'] = create_tuning_config(
#     t_minval=10, t_maxval=500, t_step=10)
#
# # bgwriter_flush_after
# # same as backend_flush_after
# PARAMS['bgwriter_flush_after']['tuning_config'] = create_tuning_config(
#     t_minval=13, t_maxval=21, t_step=0.5, t_additional_values=[0],
#     t_powers_of_2=True, t_weight_samples=True)
#
# # bgwriter_lru_maxpages
# # minval = 0, maxval = 1000, step = 50
# PARAMS['bgwriter_lru_maxpages']['tuning_config'] = create_tuning_config(
#     t_minval=0, t_maxval=1000, t_step=50)
#
# # bgwriter_lru_multiplier
# # minval = 0.0, maxval = 10.0, step = 0.5
# PARAMS['bgwriter_lru_multiplier']['tuning_config'] = create_tuning_config(
#     t_minval=0.0, t_maxval=10.0, t_step=0.5)
#
# # checkpoint_completion_target
# # minval = 0.0, maxval = 1.0, step = 0.1
# PARAMS['checkpoint_completion_target']['tuning_config'] = create_tuning_config(
#     t_minval=0.0, t_maxval=1.0, t_step=0.1)
#
# # checkpoint_flush_after
# # same as backend_flush_after
# PARAMS['checkpoint_flush_after']['tuning_config'] = create_tuning_config(
#     t_minval=13, t_maxval=21, t_step=0.5, t_additional_values=[0], t_powers_of_2=True)
#
# # checkpoint_timeout
# # minval = 5min, maxval = 3 hours, step = 5min
# # other_values = 1min (maybe)
# PARAMS['checkpoint_timeout']['tuning_config'] = create_tuning_config(
#     t_minval=300000, t_maxval=10800000, t_step=300000, t_additional_values=[60000])
#
# # commit_delay
# # minval = 0, maxval = 10000, step = 500
# PARAMS['commit_delay']['tuning_config'] = create_tuning_config(
#     t_minval=0, t_maxval=10000, t_step=500)
#
# # commit_siblings
# # minval = 0, maxval = 20, step = 1
# PARAMS['commit_siblings']['tuning_config'] = create_tuning_config(
#     t_minval=0, t_maxval=20, t_step=1)
#
# # deadlock_timeout
# # minval = 500, maxval = 20000, step = 500
# PARAMS['deadlock_timeout']['tuning_config'] = create_tuning_config(
#     t_minval=500, t_maxval=20000, t_step=500)
#
# # default_statistics_target
# # minval = 50, maxval = 2000, step = 50
# PARAMS['default_statistics_target']['tuning_config'] = create_tuning_config(
#     t_minval=50, t_maxval=2000, t_step=50)
#
# # effective_cache_size
# # eff_min = 256MB = 2^19, eff_max = over max memory (by 25%)
# # other_values = []
# # powers_of_2 = true
# PARAMS['effective_cache_size']['tuning_config'] = create_tuning_config(
#     t_minval=19, t_maxval=1.25, t_maxval_type='percentage', t_resource_type='memory',
#     t_step=0.5, t_powers_of_2=True, t_weight_samples=True,
#     t_notes='t_maxval = 25% amt greater than max memory')
#
# # effective_io_concurrency
# # minval = 0, maxval = 10, step = 1
# PARAMS['effective_io_concurrency']['tuning_config'] = create_tuning_config(
#     t_minval=0, t_maxval=10, t_step=1)
#
# # from_collapse_limit
# # minval = 4, maxval = 40, step = 4
# # other_values = 1
# PARAMS['from_collapse_limit']['tuning_config'] = create_tuning_config(
#     t_minval=4, t_maxval=40, t_step=4, t_additional_values=[1])
#
# # join_collapse_limit
# # minval = 4, maxval = 40, step = 4
# # other_values = 1
# PARAMS['join_collapse_limit']['tuning_config'] = create_tuning_config(
#     t_minval=4, t_maxval=40, t_step=4, t_additional_values=[1])
#
# # random_page_cost
# # minval = current value of seq_page_cost, maxval = seq_page_cost + 5, step = 0.5
# PARAMS['random_page_cost']['tuning_config'] = create_tuning_config(
#     t_minval=None, t_maxval=None, t_step=0.5, t_dependent=True,
#     t_notes='t_minval = current value of seq_page_cost, t_maxval = seq_page_cost + 5')
#
# # seq_page_cost
# # minval = 0.0, maxval = 2.0, step = 0.1
# PARAMS['seq_page_cost']['tuning_config'] = create_tuning_config(
#     t_minval=0.0, t_maxval=2.0, t_step=0.1)
#
# # maintenance_work_mem
# # eff_min 8MB, eff_max = 1/2 - 3/4
# PARAMS['maintenance_work_mem']['tuning_config'] = create_tuning_config(
#     t_minval=23, t_maxval=0.4, t_maxval_type='percentage', t_resource_type='memory',
#     t_step=0.5, t_powers_of_2=True, #t_weight_samples=True,
#     t_notes='t_maxval = 40% of total memory')
#
# # max_parallel_workers_per_gather
# # minval = 0, maxval = current value of max_worker_processes
# PARAMS['max_parallel_workers_per_gather']['tuning_config'] = create_tuning_config(
#     t_minval=0, t_maxval=None, t_step=1, t_dependent=True,
#     t_notes='t_maxval = max_worker_processes')
#
# # max_wal_size
# # eff_min = 2^25, eff_max = 10GB? some percentage of total disk space?
# PARAMS['max_wal_size']['tuning_config'] = create_tuning_config(
#     t_minval=25, t_maxval=33.5, t_step=0.5, t_powers_of_2=True,
#     t_weight_samples=True, t_notes='t_maxval = some % of total disk space')
#
# # max_worker_processes
# # min = 4, max = 16, step = 2
# PARAMS['max_worker_processes']['tuning_config'] = create_tuning_config(
#     t_minval=4, t_maxval=16, t_step=2)
#
# # min_parallel_relation_size
# # min = 1MB = 2^20, max = 2^30
# PARAMS['min_parallel_relation_size']['tuning_config'] = create_tuning_config(
#     t_minval=20, t_maxval=2^30, t_step=0.5, t_powers_of_2=True)
#
# # min_wal_size
# # default = 80MB, some min, then max is up to current max_wal_size
# PARAMS['min_wal_size']['tuning_config'] = create_tuning_config(
#     t_minval=25, t_maxval=None, t_step=0.5, t_powers_of_2=True,
#     t_dependent=True, t_notes='t_maxval = max_wal_size')
#
# # shared buffers
# # min = 8388608 = 2^23, max = 70% of total memory
# PARAMS['shared_buffers']['tuning_config'] = create_tuning_config(
#     t_minval=23, t_maxval=0.7, t_maxval_type='percentage', t_resource_type='memory',
#     t_step=0.5, t_powers_of_2=True, t_weight_samples=True,
#     t_notes='t_maxval = 70% of total memory')
#
# # temp buffers
# # min ~ 2^20, max = some percent of total memory
# PARAMS['temp_buffers']['tuning_config'] = create_tuning_config(
#     t_minval=20, t_maxval=0.25, t_maxval_type='percentage', t_resource_type='memory',
#     t_step=0.5, t_powers_of_2=True, t_weight_samples=True,
#     t_notes='t_maxval = some % of total memory')
#
# # wal_buffers
# # min = 32kB = 2^15, max = 2GB
# # other_values = [-1]
# PARAMS['wal_buffers']['tuning_config'] = create_tuning_config(
#     t_minval=15, t_maxval=30.5, t_step=0.5, t_powers_of_2=True,
#     t_additional_values=[-1], t_weight_samples=True)
#
# # wal_sync_method
# # enum: [open_datasync, fdatasync, fsync, open_sync]
# PARAMS['wal_sync_method']['tuning_config'] = create_tuning_config(
#     t_enumvals=['open_datasync', 'fdatasync', 'fsync', 'open_sync'])
#
# # wal_writer_delay
# # min = 50ms, max = 1000ms, step = 50ms
# # other_values = 10ms
# PARAMS['wal_writer_delay']['tuning_config'] = create_tuning_config(
#     t_minval=50, t_maxval=1000, t_step=50, t_additional_values=[10])
#
# # wal_writer_flush_after
# # same as backend_flush_after
# PARAMS['wal_writer_flush_after']['tuning_config'] = create_tuning_config(
#     t_minval=13, t_maxval=21, t_step=0.5, t_additional_values=[0], t_powers_of_2=True)
#
# # work_mem
# # min = 64kB = 2^16, max = some percent of total memory
# PARAMS['work_mem']['tuning_config'] = create_tuning_config(
#     t_minval=16, t_maxval=0.3, t_maxval_type='percentage', t_resource_type='memory',
#     t_step=0.5, t_powers_of_2=True, t_weight_samples=True, t_dependent=True,
#     t_notes='t_maxval = 30% of total memory')

# max_name_len = 0
# contexts = set()
# for pname, pinfo in PARAMS.iteritems():
#     if pinfo['tunable'] == 'yes':
#         assert pinfo['tuning_config'] is not None
#         if pinfo['unit'] == 'bytes':
#             assert pinfo['tuning_config']['t_powers_of_2'] == True
#     if len(pname) > max_name_len:
#         max_name_len = len(pname)
#     contexts.add(pinfo['context'])
# print "Max name length: {}".format(max_name_len)
# print "Contexts: {}".format(contexts)

TMP_PARAMS = OrderedDict()
for k, v in list(PARAMS.items()):
    newname = PARAM_PREFIX + '.' + k
    v['name'] = newname
    TMP_PARAMS[newname] = v
PARAMS = TMP_PARAMS

with open("settings.json", "w") as f:
    json.dump(PARAMS, f, indent=4)


# maxlen = 0
# for pname, pinfo in PARAMS.iteritems():
#     length = len(str(pinfo['default']))
#     if length > maxlen:
#         maxlen = length
#         print pname, length
# print "maxlen: {}".format(maxlen)

JSON_SETTINGS = []
SORTED_KNOB_NAMES = []
for pname, pinfo in sorted(PARAMS.items()):
    entry = {}
    entry['model'] = 'website.KnobCatalog'
    fields = dict(pinfo)
    fields['tunable'] = fields['tunable'] == 'yes'
    for k, v in list(fields.items()):
        if v is not None and not isinstance(v, str) and not isinstance(v, bool):
            fields[k] = str(v)
    fields['dbms'] = 1
    entry['fields'] = fields
    JSON_SETTINGS.append(entry)
    SORTED_KNOB_NAMES.append(pname)

with open("postgres-96_knobs.json", "w") as f:
    json.dump(JSON_SETTINGS, f, indent=4)

shutil.copy("postgres-96_knobs.json", "../../../../website/fixtures/postgres-96_knobs.json")

# sorted_knobs = [{
#     'model': 'website.PipelineResult',
#     'fields': {
#         "dbms": 1,
#         "task_type": 1,
#         "component": 4,
#         "hardware": 17,
#         "version_id": 0,
#         "value": json.dumps(SORTED_KNOB_NAMES),
#     }
# }]
# fname = 'postgres-96_sorted_knob_labels.json'
# with open(fname, "w") as f:
#     json.dump(sorted_knobs, f, indent=4)
# shutil.copy(fname, "../../../preload/")

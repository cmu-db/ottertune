import sys
import csv
import json
import shutil
from collections import OrderedDict

pg_system = [
    (1024 ** 5, 'PB'),
    (1024 ** 4, 'TB'),
    (1024 ** 3, 'GB'),
    (1024 ** 2, 'MB'),
    (1024 ** 1, 'kB'),
    (1024 ** 0, 'B'),
    ]

pg_time = [
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


STRING    = 1
INTEGER   = 2
REAL      = 3
BOOL      = 4
ENUM      = 5
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

def convert(size, system=pg_system):
    for factor, suffix in system:
        if size.endswith(suffix):
            if len(size) == len(suffix):
                amount = 1
            else:
                amount = int(size[:-len(suffix)])
            return amount * factor
    return None

params = OrderedDict()

PARAM_PREFIX = 'global'

with open("settings.csv", "r") as f:
    reader = csv.reader(f, delimiter=',')
    header = None
    for i, row in enumerate(reader):
        if i == 0:
            header = list(row)
        else:
            param = {}
            param['name'] = row[header.index('name')]
            param['vartype'] = TYPE_NAMES[row[header.index('vartype')]]
            param['category'] = row[header.index('category')]
            param['enumvals'] = row[header.index('enumvals')]

            param['context'] = row[header.index('context')]
            param['unit'] = None
            param['tunable'] = None
            param['scope'] = 'global'
            param['summary'] = row[header.index('short_desc')]
            param['description'] = row[header.index('extra_desc')]

            default = row[header.index('boot_val')]
            minval = row[header.index('min_val')]
            maxval = row[header.index('max_val')]
            if param['vartype'] == INTEGER:
                default = int(default)
                minval = int(minval)
                maxval = int(maxval)
            elif param['vartype'] == REAL:
                default = float(default)
                minval = float(minval)
                maxval = float(maxval)
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

            pg_unit = row[header.index('unit')]
            if pg_unit != '':
                factor = convert(pg_unit)
                if factor is None:
                    factor = convert(pg_unit, system=pg_time)
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

            params[param['name']] = param

# We only want to tune some settings
params['allow_system_table_mods']['tunable'] = 'no'
params['archive_mode']['tunable'] = 'no'
params['archive_timeout']['tunable'] = 'no'
params['array_nulls']['tunable'] = 'no'
params['authentication_timeout']['tunable'] = 'no'
params['backend_flush_after']['tunable'] = 'yes'
params['backslash_quote']['tunable'] = 'no'
params['bgwriter_delay']['tunable'] = 'yes'
params['bgwriter_flush_after']['tunable'] = 'yes'
params['bgwriter_lru_maxpages']['tunable'] = 'yes'
params['bgwriter_lru_multiplier']['tunable'] = 'yes'
params['bonjour']['tunable'] = 'no'
params['bonjour_name']['tunable'] = 'no'
params['bytea_output']['tunable'] = 'no'
params['check_function_bodies']['tunable'] = 'no'
params['checkpoint_completion_target']['tunable'] = 'yes'
params['checkpoint_flush_after']['tunable'] = 'yes'
params['checkpoint_timeout']['tunable'] = 'yes'
params['checkpoint_warning']['tunable'] = 'no'
params['client_min_messages']['tunable'] = 'no'
params['commit_delay']['tunable'] = 'yes'
params['commit_siblings']['tunable'] = 'yes'
params['constraint_exclusion']['tunable'] = 'no'
params['cpu_index_tuple_cost']['tunable'] = 'maybe'
params['cpu_operator_cost']['tunable'] = 'maybe'
params['cpu_tuple_cost']['tunable'] = 'maybe'
params['cursor_tuple_fraction']['tunable'] = 'maybe'
params['db_user_namespace']['tunable'] = 'no'
params['deadlock_timeout']['tunable'] = 'yes'
params['default_statistics_target']['tunable'] = 'yes'
params['default_transaction_deferrable']['tunable'] = 'no'
params['default_transaction_isolation']['tunable'] = 'no'
params['default_transaction_read_only']['tunable'] = 'no'
params['default_with_oids']['tunable'] = 'no'
params['dynamic_shared_memory_type']['tunable'] = 'no'
params['effective_cache_size']['tunable'] = 'yes'
params['effective_io_concurrency']['tunable'] = 'yes'
params['escape_string_warning']['tunable'] = 'no'
params['exit_on_error']['tunable'] = 'no'
params['extra_float_digits']['tunable'] = 'no'
params['force_parallel_mode']['tunable'] = 'no'
params['from_collapse_limit']['tunable'] = 'yes'
params['fsync']['tunable'] = 'no' # dangerous
params['full_page_writes']['tunable'] = 'no' # dangerous
params['gin_fuzzy_search_limit']['tunable'] = 'no'
params['gin_pending_list_limit']['tunable'] = 'no'
params['huge_pages']['tunable'] = 'no'
params['idle_in_transaction_session_timeout']['tunable'] = 'no'
params['ignore_checksum_failure']['tunable'] = 'no'
params['ignore_system_indexes']['tunable'] = 'no'
params['IntervalStyle']['tunable'] = 'no'
params['join_collapse_limit']['tunable'] = 'yes'
params['krb_caseins_users']['tunable'] = 'no'
params['lo_compat_privileges']['tunable'] = 'no'
params['lock_timeout']['tunable'] = 'no' # Tuning is not recommended in Postgres 9.6 manual
params['maintenance_work_mem']['tunable'] = 'yes'
params['max_connections']['tunable'] = 'no' # This is set based on # of client connections
params['max_files_per_process']['tunable'] = 'no' # Should only be increased if OS complains
params['max_locks_per_transaction']['tunable'] = 'no'
params['max_parallel_workers_per_gather']['tunable'] = 'yes' # Must be < max_worker_processes
params['max_pred_locks_per_transaction']['tunable'] = 'no'
params['max_prepared_transactions']['tunable'] = 'no'
params['max_replication_slots']['tunable'] = 'no'
params['max_stack_depth']['tunable'] = 'no'
params['max_wal_senders']['tunable'] = 'no'
params['max_wal_size']['tunable'] = 'yes'
params['max_worker_processes']['tunable'] = 'yes'
params['min_parallel_relation_size']['tunable'] = 'yes'
params['min_wal_size']['tunable'] = 'yes'
params['old_snapshot_threshold']['tunable'] = 'no'
params['operator_precedence_warning']['tunable'] = 'no'
params['parallel_setup_cost']['tunable'] = 'maybe'
params['parallel_tuple_cost']['tunable'] = 'maybe'
params['password_encryption']['tunable'] = 'no'
params['port']['tunable'] = 'no'
params['post_auth_delay']['tunable'] = 'no'
params['pre_auth_delay']['tunable'] = 'no'
params['quote_all_identifiers']['tunable'] = 'no'
params['random_page_cost']['tunable'] = 'yes'
params['replacement_sort_tuples']['tunable'] = 'no'
params['restart_after_crash']['tunable'] = 'no'
params['row_security']['tunable'] = 'no'
params['seq_page_cost']['tunable'] = 'yes'
params['session_replication_role']['tunable'] = 'no'
params['shared_buffers']['tunable'] = 'yes'
params['sql_inheritance']['tunable'] = 'no'
params['standard_conforming_strings']['tunable'] = 'no'
params['statement_timeout']['tunable'] = 'no'
params['superuser_reserved_connections']['tunable'] = 'no'
params['synchronize_seqscans']['tunable'] = 'no'
params['synchronous_commit']['tunable'] = 'no' # dangerous
params['temp_buffers']['tunable'] = 'yes'
params['temp_file_limit']['tunable'] = 'no'
params['transaction_deferrable']['tunable'] = 'no'
params['transaction_isolation']['tunable'] = 'no'
params['transaction_read_only']['tunable'] = 'no'
params['transform_null_equals']['tunable'] = 'no'
params['unix_socket_permissions']['tunable'] = 'no'
params['update_process_title']['tunable'] = 'no'
params['wal_buffers']['tunable'] = 'yes'
params['wal_compression']['tunable'] = 'no'
params['wal_keep_segments']['tunable'] = 'no'
params['wal_level']['tunable'] = 'no'
params['wal_log_hints']['tunable'] = 'no'
params['wal_sync_method']['tunable'] = 'yes'
params['wal_writer_delay']['tunable'] = 'yes'
params['wal_writer_flush_after']['tunable'] = 'yes'
params['work_mem']['tunable'] = 'yes'
params['xmlbinary']['tunable'] = 'no'
params['xmloption']['tunable'] = 'no'
params['zero_damaged_pages']['tunable'] = 'no'


with open('tunable_params.txt', 'w') as f:
    for opt in ['yes', 'maybe', 'no', '']:
        f.write(opt.upper() + '\n')
        f.write('---------------------------------------------------\n')
        for p, pdict in params.iteritems():
            if pdict['tunable'] == opt:
                f.write('{}\t{}\t{}\n'.format(p, pdict['vartype'], pdict['unit']))
        f.write('\n')

# MAX_MEM = 36  # 64GB or 2^36
#
# # backend_flush_after - range between 0 & 2MB
# # max = 2^21, eff_min = 2^13 (8kB), step either 0.5 or 1
# # other_values = [0]
# # powers_of_2 = true
# params['backend_flush_after']['tuning_config'] = create_tuning_config(
#     t_minval=13, t_maxval=21, t_step=0.5, t_additional_values=[0],
#     t_powers_of_2=True, t_weight_samples=True)
#
# # bgwriter_delay
# # true minval = 10, maxval = 500, step = 10
# params['bgwriter_delay']['tuning_config'] = create_tuning_config(
#     t_minval=10, t_maxval=500, t_step=10)
#
# # bgwriter_flush_after
# # same as backend_flush_after
# params['bgwriter_flush_after']['tuning_config'] = create_tuning_config(
#     t_minval=13, t_maxval=21, t_step=0.5, t_additional_values=[0],
#     t_powers_of_2=True, t_weight_samples=True)
#
# # bgwriter_lru_maxpages
# # minval = 0, maxval = 1000, step = 50
# params['bgwriter_lru_maxpages']['tuning_config'] = create_tuning_config(
#     t_minval=0, t_maxval=1000, t_step=50)
#
# # bgwriter_lru_multiplier
# # minval = 0.0, maxval = 10.0, step = 0.5
# params['bgwriter_lru_multiplier']['tuning_config'] = create_tuning_config(
#     t_minval=0.0, t_maxval=10.0, t_step=0.5)
#
# # checkpoint_completion_target
# # minval = 0.0, maxval = 1.0, step = 0.1
# params['checkpoint_completion_target']['tuning_config'] = create_tuning_config(
#     t_minval=0.0, t_maxval=1.0, t_step=0.1)
#
# # checkpoint_flush_after
# # same as backend_flush_after
# params['checkpoint_flush_after']['tuning_config'] = create_tuning_config(
#     t_minval=13, t_maxval=21, t_step=0.5, t_additional_values=[0], t_powers_of_2=True)
#
# # checkpoint_timeout
# # minval = 5min, maxval = 3 hours, step = 5min
# # other_values = 1min (maybe)
# params['checkpoint_timeout']['tuning_config'] = create_tuning_config(
#     t_minval=300000, t_maxval=10800000, t_step=300000, t_additional_values=[60000])
#
# # commit_delay
# # minval = 0, maxval = 10000, step = 500
# params['commit_delay']['tuning_config'] = create_tuning_config(
#     t_minval=0, t_maxval=10000, t_step=500)
#
# # commit_siblings
# # minval = 0, maxval = 20, step = 1
# params['commit_siblings']['tuning_config'] = create_tuning_config(
#     t_minval=0, t_maxval=20, t_step=1)
#
# # deadlock_timeout
# # minval = 500, maxval = 20000, step = 500
# params['deadlock_timeout']['tuning_config'] = create_tuning_config(
#     t_minval=500, t_maxval=20000, t_step=500)
#
# # default_statistics_target
# # minval = 50, maxval = 2000, step = 50
# params['default_statistics_target']['tuning_config'] = create_tuning_config(
#     t_minval=50, t_maxval=2000, t_step=50)
#
# # effective_cache_size
# # eff_min = 256MB = 2^19, eff_max = over max memory (by 25%)
# # other_values = []
# # powers_of_2 = true
# params['effective_cache_size']['tuning_config'] = create_tuning_config(
#     t_minval=19, t_maxval=1.25, t_maxval_type='percentage', t_resource_type='memory',
#     t_step=0.5, t_powers_of_2=True, t_weight_samples=True,
#     t_notes='t_maxval = 25% amt greater than max memory')
#
# # effective_io_concurrency
# # minval = 0, maxval = 10, step = 1
# params['effective_io_concurrency']['tuning_config'] = create_tuning_config(
#     t_minval=0, t_maxval=10, t_step=1)
#
# # from_collapse_limit
# # minval = 4, maxval = 40, step = 4
# # other_values = 1
# params['from_collapse_limit']['tuning_config'] = create_tuning_config(
#     t_minval=4, t_maxval=40, t_step=4, t_additional_values=[1])
#
# # join_collapse_limit
# # minval = 4, maxval = 40, step = 4
# # other_values = 1
# params['join_collapse_limit']['tuning_config'] = create_tuning_config(
#     t_minval=4, t_maxval=40, t_step=4, t_additional_values=[1])
#
# # random_page_cost
# # minval = current value of seq_page_cost, maxval = seq_page_cost + 5, step = 0.5
# params['random_page_cost']['tuning_config'] = create_tuning_config(
#     t_minval=None, t_maxval=None, t_step=0.5, t_dependent=True,
#     t_notes='t_minval = current value of seq_page_cost, t_maxval = seq_page_cost + 5')
#
# # seq_page_cost
# # minval = 0.0, maxval = 2.0, step = 0.1
# params['seq_page_cost']['tuning_config'] = create_tuning_config(
#     t_minval=0.0, t_maxval=2.0, t_step=0.1)
#
# # maintenance_work_mem
# # eff_min 8MB, eff_max = 1/2 - 3/4
# params['maintenance_work_mem']['tuning_config'] = create_tuning_config(
#     t_minval=23, t_maxval=0.4, t_maxval_type='percentage', t_resource_type='memory',
#     t_step=0.5, t_powers_of_2=True, #t_weight_samples=True,
#     t_notes='t_maxval = 40% of total memory')
#
# # max_parallel_workers_per_gather
# # minval = 0, maxval = current value of max_worker_processes
# params['max_parallel_workers_per_gather']['tuning_config'] = create_tuning_config(
#     t_minval=0, t_maxval=None, t_step=1, t_dependent=True,
#     t_notes='t_maxval = max_worker_processes')
#
# # max_wal_size
# # eff_min = 2^25, eff_max = 10GB? some percentage of total disk space?
# params['max_wal_size']['tuning_config'] = create_tuning_config(
#     t_minval=25, t_maxval=33.5, t_step=0.5, t_powers_of_2=True,
#     t_weight_samples=True, t_notes='t_maxval = some % of total disk space')
#
# # max_worker_processes
# # min = 4, max = 16, step = 2
# params['max_worker_processes']['tuning_config'] = create_tuning_config(
#     t_minval=4, t_maxval=16, t_step=2)
#
# # min_parallel_relation_size
# # min = 1MB = 2^20, max = 2^30
# params['min_parallel_relation_size']['tuning_config'] = create_tuning_config(
#     t_minval=20, t_maxval=2^30, t_step=0.5, t_powers_of_2=True)
#
# # min_wal_size
# # default = 80MB, some min, then max is up to current max_wal_size
# params['min_wal_size']['tuning_config'] = create_tuning_config(
#     t_minval=25, t_maxval=None, t_step=0.5, t_powers_of_2=True,
#     t_dependent=True, t_notes='t_maxval = max_wal_size')
#
# # shared buffers
# # min = 8388608 = 2^23, max = 70% of total memory
# params['shared_buffers']['tuning_config'] = create_tuning_config(
#     t_minval=23, t_maxval=0.7, t_maxval_type='percentage', t_resource_type='memory',
#     t_step=0.5, t_powers_of_2=True, t_weight_samples=True,
#     t_notes='t_maxval = 70% of total memory')
#
# # temp buffers
# # min ~ 2^20, max = some percent of total memory
# params['temp_buffers']['tuning_config'] = create_tuning_config(
#     t_minval=20, t_maxval=0.25, t_maxval_type='percentage', t_resource_type='memory',
#     t_step=0.5, t_powers_of_2=True, t_weight_samples=True,
#     t_notes='t_maxval = some % of total memory')
#
# # wal_buffers
# # min = 32kB = 2^15, max = 2GB
# # other_values = [-1]
# params['wal_buffers']['tuning_config'] = create_tuning_config(
#     t_minval=15, t_maxval=30.5, t_step=0.5, t_powers_of_2=True,
#     t_additional_values=[-1], t_weight_samples=True)
#
# # wal_sync_method
# # enum: [open_datasync, fdatasync, fsync, open_sync]
# params['wal_sync_method']['tuning_config'] = create_tuning_config(
#     t_enumvals=['open_datasync', 'fdatasync', 'fsync', 'open_sync'])
#
# # wal_writer_delay
# # min = 50ms, max = 1000ms, step = 50ms
# # other_values = 10ms
# params['wal_writer_delay']['tuning_config'] = create_tuning_config(
#     t_minval=50, t_maxval=1000, t_step=50, t_additional_values=[10])
#
# # wal_writer_flush_after
# # same as backend_flush_after
# params['wal_writer_flush_after']['tuning_config'] = create_tuning_config(
#     t_minval=13, t_maxval=21, t_step=0.5, t_additional_values=[0], t_powers_of_2=True)
#
# # work_mem
# # min = 64kB = 2^16, max = some percent of total memory
# params['work_mem']['tuning_config'] = create_tuning_config(
#     t_minval=16, t_maxval=0.3, t_maxval_type='percentage', t_resource_type='memory',
#     t_step=0.5, t_powers_of_2=True, t_weight_samples=True, t_dependent=True,
#     t_notes='t_maxval = 30% of total memory')

# max_name_len = 0
# contexts = set()
# for pname, pinfo in params.iteritems():
#     if pinfo['tunable'] == 'yes':
#         assert pinfo['tuning_config'] is not None
#         if pinfo['unit'] == 'bytes':
#             assert pinfo['tuning_config']['t_powers_of_2'] == True
#     if len(pname) > max_name_len:
#         max_name_len = len(pname)
#     contexts.add(pinfo['context'])
# print "Max name length: {}".format(max_name_len)
# print "Contexts: {}".format(contexts)

tmp_params = OrderedDict()
for k, v in params.iteritems():
    newname = PARAM_PREFIX + '.' + k
    v['name'] = newname
    tmp_params[newname] = v

# params = {PARAM_PREFIX + '.' + k: v['name'] for k, v in params.iteritems()}
params = tmp_params

with open("settings.json", "w") as f:
    json.dump(params, f, indent=4)


# maxlen = 0
# for pname, pinfo in params.iteritems():
#     length = len(str(pinfo['default']))
#     if length > maxlen:
#         maxlen = length
#         print pname, length
# print "maxlen: {}".format(maxlen)

json_settings = []
sorted_knob_names = []
for pname, pinfo in sorted(params.iteritems()):
    entry = {}
    entry['model'] = 'website.KnobCatalog'
    fields = dict(pinfo)
    if fields['tunable'] == 'yes':
        fields['tunable'] = True
    else:
        fields['tunable'] = False
    for k,v in fields.iteritems():
        if v is not None and not isinstance(v, str) and not isinstance(v, bool):
            fields[k] = str(v)
    fields['dbms'] = 1
    entry['fields'] = fields
    json_settings.append(entry)
    sorted_knob_names.append(pname)

with open("postgres-96_knobs.json", "w") as f:
    json.dump(json_settings, f, indent=4)

shutil.copy("postgres-96_knobs.json", "../../../preload/postgres-96_knobs.json")

#sorted_knobs = [{
#    'model': 'website.PipelineResult',
#    'fields': {
#        "dbms": 1,
#        "task_type": 1,
#        "component": 4,
#        "hardware": 17,
#        "version_id": 0,
#        "value": json.dumps(sorted_knob_names),
#    }
#}]
#
#fname = 'postgres-96_sorted_knob_labels.json'
#with open(fname, "w") as f:
#    json.dump(sorted_knobs, f, indent=4)
#
#shutil.copy(fname, "../../../preload/")



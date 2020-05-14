#
# OtterTune - create_knob_settings.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import csv
import json
import shutil
from operator import itemgetter

# Oracle Type:
# 1 - Boolean
# 2 - String
# 3 - Integer
# 4 - Parameter file
# 5 - Reserved
# 6 - Big integer


# Ottertune Type:
# STRING = 1
# INTEGER = 2
# REAL = 3
# BOOL = 4
# ENUM = 5
# TIMESTAMP = 6

# KnobResourceType
# MEMORY = 1
# CPU = 2
# STORAGE = 3
# OTHER = 4

# miss:
# OPTIMIZER_MODE
# cursor_sharing


EXTRA_KNOBS = {
    '_pga_max_size': {
        'default': 200000000,
        'resource': 1,
        'unit': 1,
    },
    '_smm_max_size': {
        'default': 100000,
        'resource': 1,
        'unit': 1,
    },
    '_smm_px_max_size': {
        'default': 300000,
        'resource': 1,
        'unit': 1,
    },
    '_optimizer_use_feedback': {
        'default': True,
        'minval': None,
        'maxval': None,
        'vartype': 4,
    },
    'ioseektim': {
        'default': 10,
        'minval': 1,
        'maxval': 10,
    },
    'iotfrspeed': {
        'default': 4096,
        'minval': 4096,
        'maxval': 190000,
    },
    '_enable_numa_optimization': {
        'default': False,
        'minval': None,
        'maxval': None,
        'vartype': 4,
    },
    '_enable_numa_support': {
        'default': False,
        'minval': None,
        'maxval': None,
        'vartype': 4,
    },
    '_unnest_subquery': {
        'default': True,
        'minval': None,
        'maxval': None,
        'vartype': 4,
    },
}


def add_fields(fields_list, version):
    for name, custom_fields in EXTRA_KNOBS.items():
        new_field = dict(
            name=('global.' + name).lower(),
            scope='global',
            dbms=version,
            category='',
            enumvals=None,
            context='',
            unit=3,  # other
            tunable=False,
            description='',
            summary='',
            vartype=2,  # integer
            minval=0,
            maxval=2000000000,
            default=500000,
        )
        new_field.update(custom_fields)
        fields_list.append(new_field)


def set_field(fields):
    if fields['name'].upper() == 'MEMORY_TARGET':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'MEMORY_MAX_TARGET':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'SGA_TARGET':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'SGA_MAX_SIZE':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'DB_CACHE_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 25000000000  # 24G
        fields['default'] = 4000000000  # 4G
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'SHARED_POOL_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 4000000000  # 4G
        fields['default'] = 1000000000  # 1G
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'SHARED_POOL_RESERVED_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 4000000000  # 4G
        fields['default'] = 1000000000  # 1G
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'SHARED_IO_POOL_SIZE':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 4000000000  # 4G
        fields['default'] = 1000000000  # 1G
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'STREAMS_POOL_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 4000000000  # 4G
        fields['default'] = 20000000  # 20M
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'LOG_BUFFER':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 2000000000  # 2GB
        fields['default'] = 50000000  # 50M
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'DB_KEEP_CACHE_SIZE':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 2000000000  # 2GB
        fields['default'] = 500000000  # 500M
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].lower() == 'db_32k_cache_size':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 2000000000  # 2GB
        fields['default'] = 500000000  # 500M
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'DB_RECYCLE_CACHE_SIZE':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 2000000000  # 2GB
        fields['default'] = 500000000  # 500M
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'LARGE_POOL_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 2000000000  # 2GB
        fields['default'] = 500000000  # 500M
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'PGA_AGGREGATE_TARGET':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].lower() == 'bitmap_merge_area_size':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 5000000000  # 3G
        fields['default'] = 0
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].lower() == 'create_bitmap_area_size':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 5000000000  # 3G
        fields['default'] = 0
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].lower() == 'hash_area_size':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 3000000000  # 3G
        fields['default'] = 0
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].lower() == 'sort_area_size':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 3000000000  # 3G
        fields['default'] = 0
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].upper() == 'OPEN_CURSORS':
        fields['tunable'] = True
        fields['minval'] = 200
        fields['maxval'] = 400
        fields['default'] = 300
    if fields['name'].upper() == 'DB_FILE_MULTIBLOCK_READ_COUNT':
        fields['tunable'] = True
        fields['minval'] = 20
        fields['maxval'] = 256
        fields['default'] = 128
    if fields['name'].upper() == 'optimizer_index_cost_adj'.upper():
        fields['tunable'] = False
        fields['minval'] = 1
        fields['maxval'] = 10000
        fields['default'] = 100
    if fields['name'].upper() == 'OPTIMIZER_USE_PENDING_STATISTICS':
        fields['tunable'] = False
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = False
        fields['vartype'] = 4
    if fields['name'].upper() == 'OPTIMIZER_USE_INVISIBLE_INDEXES':
        fields['tunable'] = False
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = False
        fields['vartype'] = 4
    if fields['name'].upper() == 'OPTIMIZER_USE_SQL_PLAN_BASELINES':
        fields['tunable'] = False
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = True
        fields['vartype'] = 4
    if fields['name'].upper() == 'OPTIMIZER_CAPTURE_SQL_PLAN_BASELINES':
        fields['tunable'] = False
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = False
        fields['vartype'] = 4
    if fields['name'].lower() == 'optimizer_dynamic_sampling':
        fields['tunable'] = True
        fields['minval'] = 2
        fields['maxval'] = 10
        fields['default'] = 2
    if fields['name'].lower() == 'optimizer_adaptive_plans':
        fields['tunable'] = True
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = True
        fields['vartype'] = 4
    if fields['name'].lower() == 'optimizer_adaptive_statistics':
        fields['tunable'] = True
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = True
        fields['vartype'] = 4
    if fields['name'].lower() == 'optimizer_features_enable':
        fields['tunable'] = True
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = '12.2.0.1'
        fields['vartype'] = 5
        fields['enumvals'] = '11.2.0.1,11.2.0.2,11.2.0.3,11.2.0.4,12.1.0.1,12.1.0.2,12.2.0.1'
    if fields['name'].lower() == 'optimizer_inmemory_aware':
        fields['tunable'] = True
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = True
        fields['vartype'] = 4
    if fields['name'].upper() == 'DISK_ASYNCH_IO':
        fields['tunable'] = True
        fields['vartype'] = 4
        fields['default'] = True
        fields['minval'] = None
        fields['maxval'] = None
    if fields['name'].lower() == 'db_writer_processes':
        fields['tunable'] = False
        fields['minval'] = 1
        fields['maxval'] = 10
        fields['default'] = 1
    if fields['name'].lower() == 'filesystemio_options':
        fields['default'] = 'none'
        fields['minval'] = None
        fields['maxval'] = None
        fields['vartype'] = 5
        fields['enumvals'] = 'asynch,directio,none,setall'
    if fields['name'].lower() == 'optimizer_mode':
        fields['default'] = 'ALL_ROWS'
        fields['minval'] = None
        fields['maxval'] = None
        fields['vartype'] = 5
        fields['enumvals'] = 'FIRST_ROWS_1,FIRST_ROWS_10,FIRST_ROWS_100,FIRST_ROWS_1000,FIRST_ROWS,ALL_ROWS,CHOOSE'
    if fields['name'].lower() == 'workarea_size_policy':
        fields['default'] = 'AUTO'
        fields['minval'] = None
        fields['maxval'] = None
        fields['vartype'] = 5
        fields['enumvals'] = 'AUTO,MANUAL'
    if fields['name'].lower() == 'cursor_sharing':
        fields['default'] = 'EXACT'
        fields['minval'] = None
        fields['maxval'] = None
        fields['vartype'] = 5
        fields['enumvals'] = 'SIMILAR,EXACT,FORCE'
    if fields['name'].lower() == 'java_jit_enabled':
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = True
        fields['vartype'] = 4
    if fields['name'].lower() == 'java_pool_size':
        fields['minval'] = 0
        fields['maxval'] = 1000000000
        fields['default'] = 0
        fields['resource'] = 1
        fields['unit'] = 1
    if fields['name'].lower() == 'log_archive_max_processes':
        fields['minval'] = 1
        fields['maxval'] = 30
        fields['default'] = 4
    if fields['name'].lower() == 'commit_logging':
        fields['default'] = ''
        fields['minval'] = None
        fields['maxval'] = None
        fields['vartype'] = 5
        fields['enumvals'] = ',IMMEDIATE,BATCH'
    if fields['name'].lower() == 'result_cache_max_result':
        fields['minval'] = 0
        fields['maxval'] = 100
        fields['default'] = 5
    if fields['name'].lower() == 'approx_for_aggregation':
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = False
        fields['vartype'] = 4
    if fields['name'].lower() == 'approx_for_count_distinct':
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = False
        fields['vartype'] = 4
    if fields['name'].lower() == 'approx_for_percentile':
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = 'NONE'
        fields['vartype'] = 5
        fields['enumvals'] = 'NONE,PERCENTILE_CONT,PERCENTILE_CONT DETERMINISTIC,PERCENTILE_DISC,PERCENTILE_DISC DETERMINISTIC,ALL,ALL DETERMINISTIC'
    if fields['name'].lower() == 'session_cached_cursors':
        fields['minval'] = 0
        fields['maxval'] = 100
        fields['default'] = 50
    if fields['name'].lower() == 'use_large_pages':
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = 'TRUE'
        fields['vartype'] = 5
        fields['enumvals'] = 'TRUE,FALSE,ONLY'
    if fields['name'].lower() == 'hs_autoregister':
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = True
        fields['vartype'] = 4
    if fields['name'].lower() == 'cursor_invalidation':
        fields['default'] = 'IMMEDIATE'
        fields['minval'] = None
        fields['maxval'] = None
        fields['vartype'] = 5
        fields['enumvals'] = 'IMMEDIATE,DEFERRED'
    if fields['name'].lower() == 'plsql_optimize_level':
        fields['minval'] = 0
        fields['maxval'] = 3
        fields['default'] = 2
    if fields['name'].lower() == 'db_big_table_cache_percent_target':
        fields['minval'] = 0
        fields['maxval'] = 90
        fields['default'] = 0
        fields['vartype'] = 2
    if fields['name'].lower() == 'query_rewrite_enabled':
        fields['default'] = 'TRUE'
        fields['minval'] = None
        fields['maxval'] = None
        fields['vartype'] = 5
        fields['enumvals'] = 'TRUE,FALSE,FORCE'


COLNAMES = ("NAME", "TYPE", "DEFAULT_VALUE", "DESCRIPTION")


def process_version(version, delim=','):
    fields_list = []
    add_fields(fields_list, version)
    with open('oracle{}.csv'.format(version), 'r', newline='') as f:
        reader = csv.reader(f, delimiter=delim)
        header = [h.upper() for h in next(reader)]
        idxs = [header.index(c) for c in COLNAMES]
        ncols = len(header)

        ri = 0
        for row in reader:
            assert ncols == len(row), (ri, ncols, len(row))
            fields = {}
            for i, cname in zip(idxs, COLNAMES):
                value = row[i]
                if isinstance(value, str):
                    value = value.strip()
                if cname == 'NAME':
                    fields['name'] = value.upper()
                elif cname == 'TYPE':
                    value = int(value)
                    if value == 1:
                        fields['vartype'] = 4  # Boolean
                    elif value in (3, 6):
                        fields['vartype'] = 2  # Integer
                    else:
                        fields['vartype'] = 1  # Assume it's a sting otherwise
                elif cname == 'DEFAULT_VALUE':
                    fields['default'] = value
                else:
                    fields['summary'] = value

                fields.update(
                    scope='global',
                    dbms=version,
                    category='',
                    enumvals=None,
                    context='',
                    unit=3,  # Other
                    tunable=False,
                    description='',
                    minval=None,
                    maxval=None,
                )

            set_field(fields)
            fields['name'] = ('global.' + fields['name']).lower()
            fields_list.append(fields)
            ri += 1

    fields_list = sorted(fields_list, key=itemgetter('name'))
    final_metrics = [dict(model='website.KnobCatalog', fields=fs) for fs in fields_list]
    filename = 'oracle-{}_knobs.json'.format(version)
    with open(filename, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    shutil.copy(filename, "../../../../website/fixtures/{}".format(filename))


def main():
    process_version(19)              # v19c
    process_version(12)              # v12.2c
    process_version(121, delim='|')  # v12.1c


if __name__ == '__main__':
    main()

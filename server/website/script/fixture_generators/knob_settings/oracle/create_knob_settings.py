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


def set_field(fields):
    if fields['name'].upper() == 'MEMORY_TARGET':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
        fields['resource'] = 1
    if fields['name'].upper() == 'MEMORY_MAX_TARGET':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
        fields['resource'] = 1
    if fields['name'].upper() == 'SGA_TARGET':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
        fields['resource'] = 1
    if fields['name'].upper() == 'SGA_MAX_SIZE':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
        fields['resource'] = 1
    if fields['name'].upper() == 'DB_CACHE_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 25000000000  # 24G
        fields['default'] = 4000000000  # 4G
        fields['resource'] = 1
    if fields['name'].upper() == 'SHARED_POOL_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 4000000000  # 4G
        fields['default'] = 1000000000  # 1G
        fields['resource'] = 1
    if fields['name'].upper() == 'SHARED_POOL_RESERVED_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 4000000000  # 4G
        fields['default'] = 1000000000  # 1G
        fields['resource'] = 1
    if fields['name'].upper() == 'SHARED_IO_POOL_SIZE':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 4000000000  # 4G
        fields['default'] = 1000000000  # 1G
        fields['resource'] = 1
    if fields['name'].upper() == 'STREAMS_POOL_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 4000000000  # 4G
        fields['default'] = 20000000  # 20M
        fields['resource'] = 1
    if fields['name'].upper() == 'LOG_BUFFER':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 2000000000  # 2GB
        fields['default'] = 50000000  # 50M
        fields['resource'] = 1
    if fields['name'].upper() == 'DB_KEEP_CACHE_SIZE':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 2000000000  # 2GB
        fields['default'] = 500000000  # 500M
        fields['resource'] = 1
    if fields['name'].upper() == 'DB_RECYCLE_CACHE_SIZE':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 2000000000  # 2GB
        fields['default'] = 500000000  # 500M
        fields['resource'] = 1
    if fields['name'].upper() == 'LARGE_POOL_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 2000000000  # 2GB
        fields['default'] = 500000000  # 500M
        fields['resource'] = 1
    if fields['name'].upper() == 'PGA_AGGREGATE_TARGET':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
        fields['resource'] = 1
    if fields['name'].lower() == 'bitmap_merge_area_size':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 5000000000  # 3G
        fields['default'] = 0
        fields['resource'] = 1
    if fields['name'].lower() == 'create_bitmap_area_size':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 5000000000  # 3G
        fields['default'] = 0
        fields['resource'] = 1
    if fields['name'].lower() == 'hash_area_size':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 3000000000  # 3G
        fields['default'] = 0
        fields['resource'] = 1
    if fields['name'].lower() == 'sort_area_size':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 3000000000  # 3G
        fields['default'] = 0
        fields['resource'] = 1
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
    if fields['name'].upper() == 'OPTIMIZER_USE_INVISIBLE_INDEXES':
        fields['tunable'] = False
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = False
    if fields['name'].upper() == 'OPTIMIZER_USE_SQL_PLAN_BASELINES':
        fields['tunable'] = False
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = True
    if fields['name'].upper() == 'OPTIMIZER_CAPTURE_SQL_PLAN_BASELINES':
        fields['tunable'] = False
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = False
    if fields['name'].lower() == 'optimizer_dynamic_sampling':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 11
        fields['default'] = 2
    if fields['name'].lower() == 'optimizer_adaptive_plans':
        fields['tunable'] = True
        fields['minval'] = None
        fields['maxval'] = None
        fields['default'] = True
    if fields['name'].upper() == 'DISK_ASYNCH_IO':
        fields['tunable'] = True
        fields['vartype'] = 5
        fields['enumvals'] = 'TRUE,FALSE'
        fields['default'] = 'TRUE'


COLNAMES = ("NAME", "TYPE", "DEFAULT_VALUE", "DESCRIPTION")


def process_version(version, delim=','):
    fields_list = []
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

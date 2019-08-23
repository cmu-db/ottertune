#
# OtterTune - create_knob_settings.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import json
import shutil

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

# miss:
# OPTIMIZER_MODE
# cursor_sharing


def set_field(fields):
    if fields['name'].upper() == 'MEMORY_TARGET':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
    if fields['name'].upper() == 'MEMORY_MAX_TARGET':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
    if fields['name'].upper() == 'SGA_TARGET':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
    if fields['name'].upper() == 'SGA_MAX_SIZE':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
    if fields['name'].upper() == 'DB_CACHE_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 25000000000  # 24G
        fields['default'] = 4000000000  # 4G
    if fields['name'].upper() == 'SHARED_POOL_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 4000000000  # 4G
        fields['default'] = 1000000000  # 1G
    if fields['name'].upper() == 'SHARED_IO_POOL_SIZE':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 4000000000  # 4G
        fields['default'] = 1000000000  # 1G
    if fields['name'].upper() == 'STREAMS_POOL_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 4000000000  # 4G
        fields['default'] = 20000000  # 20M
    if fields['name'].upper() == 'LOG_BUFFER':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 2000000000  # 2GB
        fields['default'] = 50000000  # 50M
    if fields['name'].upper() == 'DB_KEEP_CACHE_SIZE':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 2000000000  # 2GB
        fields['default'] = 500000000  # 500M
    if fields['name'].upper() == 'DB_RECYCLE_CACHE_SIZE':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 2000000000  # 2GB
        fields['default'] = 500000000  # 500M
    if fields['name'].upper() == 'LARGE_POOL_SIZE':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 2000000000  # 2GB
        fields['default'] = 500000000  # 500M
    if fields['name'].upper() == 'PGA_AGGREGATE_TARGET':
        fields['tunable'] = False
        fields['minval'] = 0
        fields['maxval'] = 33000000000  # 33G
        fields['default'] = 0
    if fields['name'].lower() == 'bitmap_merge_area_size':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 5000000000  # 3G
        fields['default'] = 0
    if fields['name'].lower() == 'create_bitmap_area_size':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 5000000000  # 3G
        fields['default'] = 0
    if fields['name'].lower() == 'hash_area_size':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 3000000000  # 3G
        fields['default'] = 0
    if fields['name'].lower() == 'sort_area_size':
        fields['tunable'] = True
        fields['minval'] = 0
        fields['maxval'] = 3000000000  # 3G
        fields['default'] = 0
    if fields['name'].upper() == 'OPEN_CURSORS':
        fields['tunable'] = False
        fields['minval'] = 200
        fields['maxval'] = 400
        fields['default'] = 300
    if fields['name'].upper() == 'DB_FILE_MULTIBLOCK_READ_COUNT':
        fields['tunable'] = False
        fields['minval'] = 64
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
    if fields['name'].upper() == 'DISK_ASYNCH_IO':
        fields['tunable'] = True
        fields['vartype'] = 5
        fields['enumvals'] = 'TRUE,FALSE'
        fields['default'] = 'TRUE'


def main():
    final_metrics = []
    with open('oracle.txt', 'r') as f:
        num = 0

        lines = f.readlines()
        for line in lines:
            line = line.strip().replace("\n", "")
            if not line:
                continue
            if line in ['DESCRIPTION', 'NAME', 'TYPE'] or line.startswith('-'):
                continue
            if num == 0:
                entry = {}
                entry['model'] = 'website.KnobCatalog'
                fields = {}
                fields['name'] = line
            elif num == 1:
                if line in ['3', '6']:
                    fields['vartype'] = 2
                    fields['default'] = 0
                elif line == '1':
                    fields['vartype'] = 4
                    fields['default'] = False
                else:
                    fields['vartype'] = 1
                    fields['default'] = ''
            elif num == 2:
                fields['summary'] = line
                fields['scope'] = 'global'
                fields['dbms'] = 18       # oracle
                fields['category'] = ''
                fields['enumvals'] = None
                fields['context'] = ''
                fields['unit'] = 3       # other
                fields['tunable'] = False
                fields['scope'] = 'global'
                fields['description'] = ''
                fields['minval'] = None
                fields['maxval'] = None
                set_field(fields)
                fields['name'] = 'global.' + fields['name']
                entry['fields'] = fields
                final_metrics.append(entry)
            num = (num + 1) % 3
    with open('oracle_knobs.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
    shutil.copy("oracle_knobs.json", "../../../../website/fixtures/oracle_knobs.json")


if __name__ == '__main__':
    main()

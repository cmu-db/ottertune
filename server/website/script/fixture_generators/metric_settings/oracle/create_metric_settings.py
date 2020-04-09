#
# OtterTune - create_metric_settings.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import json
import os
import shutil
import sys
from collections import OrderedDict


# Metric catalog fields:
# dbms
# name
# vartype
# summary
# scope
# metric_type

# Ottertune Type:
# STRING = 1
# INTEGER = 2
# REAL = 3
# BOOL = 4
# ENUM = 5
# TIMESTAMP = 6

# Metric Types:
# COUNTER = 1
# INFO = 2
# STATISTICS = 3


def check_type(value):
    # if value is not None:
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    if isinstance(value, int):
        vtype = 2  # Integer
    elif isinstance(value, float):
        vtype = 3  # Real
    else:
        vtype = 1  # String

    return vtype

def create_settings(metric_data, dbms):
    metrics = []
    for name, value in metric_data.items():
        vartype = check_type(value)

        if vartype in (2, 3):  # Numeric (integer/real)
            if 'average' in name or name.endswith('current') or \
                    name.startswith('sysstat.session pga memory') or \
                    name.startswith('sysstat.session uga memory') or \
                    name.endswith('wait_class#') or \
                    name.endswith('wait_class_id'):
                mettype = 3  # Statistic
            else:
                mettype = 1  # Counter - most common type of numeric metric
        else:
            mettype = 2  # Info (anything that's not numeric)
        summary = '{}: {}'.format(name, value)

        if name == 'sysstat.user commits':
            assert vartype == 2 and 1  # Check it's an int/counter

        default = '' if len(str(value)) > 31 else value
        entry = OrderedDict([
            ('dbms', dbms),
            ('name', 'global.{}'.format(name)),
            ('vartype', vartype),
            ('default', default),
            ('summary', summary),
            ('scope', 'global'),
            ('metric_type', mettype),
        ])
        metrics.append(OrderedDict([('fields', entry), ('model', 'website.MetricCatalog')]))

    return metrics


# Versions 12.1c, 12.2c, and 19c
VERSIONS = (121, 12, 19)


def usage():
    print('python3 create_metric_settings.py [version] (valid versions: {})'.format(
        ', '.join(VERSIONS)))
    sys.exit(1)


def main():
    if len(sys.argv) == 1:
        versions = VERSIONS
    else:
        version = int(sys.argv[1])
        if version not in VERSIONS:
            usage()
        versions = (version,)

    for version in versions:
        with open('oracle{}.json'.format(version), 'r') as f:
            metrics = json.load(f, object_pairs_hook=OrderedDict)

        metrics = metrics['global']['global']
        meta = create_settings(metrics, version)
        filename = 'oracle-{}_metrics.json'.format(version)
        with open(filename, 'w') as f:
            json.dump(meta, f, indent=4)
        shutil.copy(filename, "../../../../website/fixtures/{}".format(filename))


if __name__ == '__main__':
    main()

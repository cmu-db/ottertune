#
# OtterTune - create_metric_settings.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import json
import os
import sys
from collections import OrderedDict

ROOT = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
while os.path.basename(ROOT) != 'website':
    ROOT = os.path.dirname(ROOT)
print('WEBSITE ROOT: {}'.format(ROOT))
sys.path.insert(0, ROOT)

from website.types import MetricType, VarType

# Metric catalog fields:
# dbms
# name
# vartype
# summary
# scope
# metric_type

# Constants
MODEL = 'website.MetricCatalog'
SCOPE = 'global'
VERSIONS = (12, 19)


# def main():
#     final_metrics = []
#     with open('oracle12.txt', 'r') as f:
#         odd = 0
#         entry = {}
#         fields = {}
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip().replace("\n", "")
#             if not line:
#                 continue
#             if line == 'NAME' or line.startswith('-'):
#                 continue
#             if odd == 0:
#                 entry = {}
#                 entry['model'] = 'website.MetricCatalog'
#                 fields = {}
#                 fields['name'] = "global." + line
#                 fields['summary'] = line
#                 fields['vartype'] = 2	 # int
#                 fields['scope'] = 'global'
#                 fields['metric_type'] = 3	 # stat
#                 if fields['name'] == "global.user commits":
#                     fields['metric_type'] = 1	 # counter
#                 fields['dbms'] = 12  # oracle
#                 entry['fields'] = fields
#                 final_metrics.append(entry)
#     with open('oracle-12_metrics.json', 'w') as f:
#         json.dump(final_metrics, f, indent=4)
#     shutil.copy('oracle-12_metrics.json', '../../../../website/fixtures/oracle-12_metrics.json')


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
        vtype = VarType.INTEGER
    elif isinstance(value, float):
        vtype = VarType.REAL
    else:
        vtype = VarType.STRING

    return vtype


def create_settings(metric_data, dbms):
    metrics = []
    for name, value in metric_data.items():
        vartype = check_type(value)

        if vartype in (VarType.INTEGER, VarType.REAL):
            if 'average' in name or name.endswith('current') or \
                    name.startswith('session pga memory'):
                mettype = MetricType.STATISTICS
            else:
                mettype = MetricType.COUNTER  # Most int/float metrics are counters
        else:
            mettype = MetricType.INFO
        summary = '{}: {}'.format(name, value)

        if name == 'user commits':
            assert vartype == VarType.INTEGER and mettype == MetricType.COUNTER

        entry = OrderedDict([
            ('dbms', dbms),
            ('name', 'global.{}'.format(name)),
            ('vartype', vartype),
            ('summary', summary),
            ('scope', 'global'),
            ('metric_type', mettype),
        ])
        metrics.append(OrderedDict([('fields', entry), ('model', MODEL)]))

    return metrics


def usage():
    print('python3 create_metric_settings.py [version] (valid versions: 12, 19)')
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
        savepath = os.path.join(
            ROOT, 'website', 'fixtures', 'oracle-{}_metrics.json'.format(version))
        with open(savepath, 'w') as f:
            json.dump(meta, f, indent=4)


if __name__ == '__main__':
    main()

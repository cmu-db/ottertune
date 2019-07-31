#
# OtterTune - create_metric_settings.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import json
import shutil


def main():
    final_metrics = []
    with open('oracle.txt', 'r') as f:
        odd = 0
        entry = {}
        fields = {}
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace("\n", "")
            if not line:
                continue
            if line == 'NAME' or line.startswith('-'):
                continue
            if odd == 0:
                entry = {}
                entry['model'] = 'website.MetricCatalog'
                fields = {}
                fields['name'] = "global." + line
                fields['summary'] = line
                fields['vartype'] = 2	 # int
                fields['scope'] = 'global'
                fields['metric_type'] = 3	 # stat
                if fields['name'] == "global.user commits":
                    fields['metric_type'] = 1	 # counter
                fields['dbms'] = 18  # oracle
                entry['fields'] = fields
                final_metrics.append(entry)
    with open('oracle_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
    shutil.copy('oracle_metrics.json', '../../../../website/fixtures/oracle_metrics.json')


if __name__ == '__main__':
    main()

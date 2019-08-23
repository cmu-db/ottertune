#
# OtterTune - create_metric_settings.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import json
import shutil

COUNTER = 1
INFO = 2

INTEGER = 2
STRING = 1
TIMESTAMP = 6
NUMERIC_TYPES = ['oid', 'bigint', 'double precision', 'integer']


def load_data(filename):
    with open(filename, 'r') as f:
        csv_stats = f.readlines()
    header = csv_stats[0].strip().split(',')
    stats_dict = {}
    for line in csv_stats[1:]:
        parts = line.strip().split(',', 3)
        assert len(parts) == 4, "parts: {}".format(parts)
        stat = {}
        stat['name'] = parts[header.index('column_name')]
        stat['summary'] = parts[header.index('description')]
        stat['metric_type'] = parts[header.index('metric_type')]
        vartype = parts[header.index('data_type')]
        if vartype in NUMERIC_TYPES:
            vartype = INTEGER
        elif vartype == 'name' or vartype == 'text':
            vartype = STRING
        elif vartype.startswith('timestamp'):
            vartype = TIMESTAMP
        else:
            raise Exception(vartype)
        stat['vartype'] = vartype
        stats_dict[stat['name']] = stat
    return stats_dict


def main():
    dbstats = load_data('pg96_database_stats.csv')
    gstats = load_data('pg96_global_stats.csv')
    istats = load_data('pg96_index_stats.csv')
    tstats = load_data('pg96_table_stats.csv')

    with open('metrics_sample.json', 'r') as f:
        metrics = json.load(f)

    final_metrics = []
    numeric_metric_names = []
    vartypes = set()
    for view_name, mets in sorted(metrics.items()):
        if 'database' in view_name:
            scope = 'database'
            stats = dbstats
        elif 'indexes' in view_name:
            scope = 'index'
            stats = istats
        elif 'tables' in view_name:
            scope = 'table'
            stats = tstats
        else:
            scope = 'global'
            stats = gstats

        for metric_name in mets:
            entry = {}
            entry['model'] = 'website.MetricCatalog'
            mstats = stats[metric_name]
            fields = {}
            fields['name'] = '{}.{}'.format(view_name, metric_name)
            fields['vartype'] = mstats['vartype']
            vartypes.add(fields['vartype'])
            fields['summary'] = mstats['summary']
            fields['scope'] = scope
            metric_type = mstats['metric_type']
            if metric_type == 'counter':
                numeric_metric_names.append(fields['name'])
                mt = COUNTER
            elif metric_type == 'info':
                mt = INFO
            else:
                raise Exception('Invalid metric type: {}'.format(metric_type))
            fields['metric_type'] = mt
            fields['dbms'] = 1
            entry['fields'] = fields
            final_metrics.append(entry)
    #         sorted_metric_names.append(fields['name'])

    with open('postgres-96_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)

    shutil.copy('postgres-96_metrics.json', '../../../../website/fixtures/postgres-96_metrics.json')

    with open('postgres-96_numeric_metric_names.json', 'w') as f:
        json.dump(numeric_metric_names, f, indent=4)

    # sorted_metrics = [{
    #     'model': 'website.PipelineResult',
    #     'fields': {
    #         "dbms": 1,
    #         "task_type": 2,
    #         "component": 4,
    #         "hardware": 17,
    #         "version_id": 0,
    #         "value": json.dumps(sorted_metric_names),
    #     }
    # }]
    # fname = 'postgres-96_sorted_metric_labels.json'
    # with open(fname, 'w') as f:
    #     json.dump(sorted_metrics, f, indent=4)
    # shutil.copy(fname, '../../../preload/')


if __name__ == '__main__':
    main()

#
# OtterTune - create_pruned_metrics.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import os
import shutil
import json
import itertools

DATADIR = '/dataset/oltpbench/first_paper_experiments/analysis/workload_characterization'
CLUSTERS_FNAME = 'DetK_optimal_num_clusters.txt'

DBMSS = {'postgres-9.6': 1}
HARDWARES = {'m3.xlarge': 16}
TIMESTAMP = '2016-12-04 11:00'
CONVERT = True
TASK_TYPE = 1

MODEL = 'website.PipelineResult'

SUMMARY_MAP = {
    'throughput_req_per_sec': 'Throughput (requests/second)',
    '99th_lat_ms': '99th Percentile Latency (microseconds)',
    'max_lat_ms': 'Maximum Latency (microseconds)',
}


def load_postgres_metrics():
    with open('/dataset/oltpbench/first_paper_experiments/samples/sample.metrics', 'r') as f:
        sample = json.load(f)
        metric_map = {}
        for query_name, entries in list(sample.items()):
            assert len(entries) > 0
            columns = list(entries[0].keys())
            for column in columns:
                if column not in metric_map:
                    metric_map[column] = []
                metric_map[column].append(query_name)
    return metric_map


def main():
    for dbms, hw in itertools.product(list(DBMSS.keys()), HARDWARES):
        datapath = os.path.join(DATADIR, '{}_{}'.format(dbms, hw))
        if not os.path.exists(datapath):
            raise IOError('Path does not exist: {}'.format(datapath))
        with open(os.path.join(datapath, CLUSTERS_FNAME), 'r') as f:
            num_clusters = int(f.read().strip())
        with open(os.path.join(datapath, 'featured_metrics_{}.txt'.format(num_clusters)), 'r') as f:
            mets = [p.strip() for p in f.read().split('\n')]
        if CONVERT:
            if dbms.startswith('postgres'):
                metric_map = load_postgres_metrics()
                pruned_metrics = []
                for met in mets:
                    if met in SUMMARY_MAP:
                        pruned_metrics.append(SUMMARY_MAP[met])
                    else:
                        if met not in metric_map:
                            raise Exception('Unknown metric: {}'.format(met))
                        qnames = metric_map[met]
                        assert len(qnames) > 0
                        if len(qnames) > 1:
                            raise Exception(
                                '2+ queries have the same column name: {} ({})'.format(
                                    met, qnames))
                        pruned_metrics.append('{}.{}'.format(qnames[0], met))
            else:
                raise NotImplementedError("Implement me!")
        else:
            pruned_metrics = mets
        pruned_metrics = sorted(pruned_metrics)

        basename = '{}_{}_pruned_metrics'.format(dbms, hw).replace('.', '')
        with open(basename + '.txt', 'w') as f:
            f.write('\n'.join(pruned_metrics))

        django_entry = [{
            'model': MODEL,
            'fields': {
                'dbms': DBMSS[dbms],
                'hardware': HARDWARES[hw],
                'creation_timestamp': TIMESTAMP,
                'task_type': TASK_TYPE,
                'value': json.dumps(pruned_metrics, indent=4)
            }
        }]
        savepath = basename + '.json'
        with open(savepath, 'w') as f:
            json.dump(django_entry, f, indent=4)

        shutil.copy(savepath, '../../preload/{}'.format(savepath))


if __name__ == '__main__':
    main()

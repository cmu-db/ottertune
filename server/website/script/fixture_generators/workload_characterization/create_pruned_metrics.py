import sys, os.path
import shutil
import json
import itertools
from django.utils.timezone import now

datadir = '/dataset/oltpbench/first_paper_experiments/analysis/workload_characterization'
clusters_fname = 'DetK_optimal_num_clusters.txt'

dbmss = {'postgres-9.6': 1}
hardwares = {'m3.xlarge': 16}
#ts = '"2011-09-01T13:20:30+03:00"'#datetime.datetime.now()
ts = '2016-12-04 11:00'
convert = True
task_type = 1

model = 'website.PipelineResult'

summary_map = {
    'throughput_req_per_sec': 'Throughput (requests/second)',
    '99th_lat_ms': '99th Percentile Latency (microseconds)',
    'max_lat_ms': 'Maximum Latency (microseconds)',
}

def load_postgres_metrics():
    with open('/dataset/oltpbench/first_paper_experiments/samples/sample.metrics', 'r') as f:
        sample = json.load(f)
        metric_map = {}
        for query_name, entries in sample.iteritems():
            assert len(entries) > 0
            columns = entries[0].keys()
            for column in columns:
                if column not in metric_map:
                    metric_map[column] = []
                metric_map[column].append(query_name)
    return metric_map


for dbms, hw in itertools.product(dbmss.keys(), hardwares):
    datapath = os.path.join(datadir, '{}_{}'.format(dbms, hw))
    if not os.path.exists(datapath):
        raise IOError('Path does not exist: {}'.format(datapath))
    with open(os.path.join(datapath, clusters_fname), 'r') as f:
        num_clusters = int(f.read().strip())
    with open(os.path.join(datapath, 'featured_metrics_{}.txt'.format(num_clusters)), 'r') as f:
        mets = [p.strip() for p in f.read().split('\n')]
    if convert:
        if dbms.startswith('postgres'):
            metric_map = load_postgres_metrics()
            pruned_metrics = []
            for met in mets:
                if met in summary_map:
                    pruned_metrics.append(summary_map[met])
                else:
                    if met not in metric_map:
                        raise Exception('Unknown metric: {}'.format(met))
                    qnames = metric_map[met]
                    assert len(qnames) > 0
                    if len(qnames) > 1:
                        raise Exception('2+ queries have the same column name: {} ({})'.format(met, qnames))
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
        'model': model,
        'fields': {
            'dbms': dbmss[dbms],
            'hardware': hardwares[hw],
            'creation_timestamp': ts,
            'task_type': task_type,
            'value': json.dumps(pruned_metrics, indent=4)
        }
    }]
    savepath = basename + '.json'
    with open(savepath, 'w') as f:
        json.dump(django_entry, f, indent=4)

    shutil.copy(savepath, '../../preload/{}'.format(savepath))




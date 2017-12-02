import sys, os.path
import shutil
import json
import itertools

#from django.utils.timezone import now
#from datetime.datetime import now
import datetime

datadir = '/dataset/oltpbench/first_paper_experiments/analysis/knob_identification'

dbmss = {'postgres-9.6': 1}
hardwares = {'m3.xlarge': 16}
#ts = '"2011-09-01T13:20:30+03:00"'#datetime.datetime.now()
ts = '2016-12-04 11:00'
convert = True
task_type = 2

model = 'website.PipelineResult'
validate = True
extra_exceptions = {
    'checkpoint_segments',
}

def validate_postgres(knobs, dbms):
    with open('../knob_settings/{}/{}_knobs.json'.format(dbms.replace('-', '_'), dbms.replace('.', '')), 'r') as f:
        knob_info = json.load(f)
        knob_info = {k['fields']['name']: k['fields'] for k in knob_info}
    for kname, kinfo in knob_info.iteritems():
        if not kname in knobs and kinfo['tunable'] == True:
            knobs.append(kname)
            print "WARNING: adding missing knob to end ({})".format(kname)
    knob_names = knob_info.keys()
    for kname in knobs:
        if kname not in knob_names:
            if kname not in extra_exceptions:
                raise Exception('Extra knob: {}'.format(kname))
            knobs.remove(kname)
            print "WARNING: removing extra knob ({})".format(kname)

for dbms, hw in itertools.product(dbmss.keys(), hardwares):
    datapath = os.path.join(datadir, '{}_{}'.format(dbms, hw))
    if not os.path.exists(datapath):
        raise IOError('Path does not exist: {}'.format(datapath))
    with open(os.path.join(datapath, 'featured_knobs.txt'), 'r') as f:
        knobs = [k.strip() for k in f.read().split('\n')]
    if validate and dbms.startswith('postgres'):
        validate_postgres(knobs, dbms)

    basename = '{}_{}_ranked_knobs'.format(dbms, hw).replace('.', '')
    with open(basename + '.txt', 'w') as f:
        f.write('\n'.join(knobs))

    django_entry = [{
        'model': model,
        'fields': {
            'dbms': dbmss[dbms],
            'hardware': hardwares[hw],
            'creation_timestamp': ts,
            'task_type': task_type,
            'value': json.dumps(knobs, indent=4)
        }
    }]
    savepath = basename + '.json'
    with open(savepath, 'w') as f:
        json.dump(django_entry, f, indent=4)

    shutil.copy(savepath, '../../preload/{}'.format(savepath))




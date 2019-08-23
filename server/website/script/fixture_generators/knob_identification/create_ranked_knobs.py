#
# OtterTune - create_ranked_knobs.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import logging
import os
import shutil
import json
import itertools

LOG = logging.getLogger(__name__)

DATADIR = '/dataset/oltpbench/first_paper_experiments/analysis/knob_identification'

DBMSS = {'postgres-9.6': 1}
HARDWARES = {'m3.xlarge': 16}
TIMESTAMP = '2016-12-04 11:00'
TASK_TYPE = 2

PREFIX = 'global'
MODEL = 'website.PipelineResult'
VALIDATE = True
EXTRA_EXCEPTIONS = {
    PREFIX + '.' + 'checkpoint_segments',
}


def validate_postgres(knobs, dbms):
    with open('../knob_settings/{}/{}_knobs.json'.format(dbms.replace('-', '_'),
                                                         dbms.replace('.', '')), 'r') as f:
        knob_info = json.load(f)
        knob_info = {k['fields']['name']: k['fields'] for k in knob_info}
    for kname, kinfo in list(knob_info.items()):
        if kname not in knobs and kinfo['tunable'] is True:
            knobs.append(kname)
            LOG.warning("Adding missing knob to end of list (%s)", kname)
    knob_names = list(knob_info.keys())
    for kname in knobs:
        if kname not in knob_names:
            if kname not in EXTRA_EXCEPTIONS:
                raise Exception('Extra knob: {}'.format(kname))
            knobs.remove(kname)
            LOG.warning("Removing extra knob (%s)", kname)


def main():
    for dbms, hw in itertools.product(list(DBMSS.keys()), HARDWARES):
        datapath = os.path.join(DATADIR, '{}_{}'.format(dbms, hw))
        if not os.path.exists(datapath):
            raise IOError('Path does not exist: {}'.format(datapath))
        with open(os.path.join(datapath, 'featured_knobs.txt'), 'r') as f:
            knobs = [k.strip() for k in f.read().split('\n')]
        knobs = [PREFIX + '.' + k for k in knobs]
        if VALIDATE and dbms.startswith('postgres'):
            validate_postgres(knobs, dbms)

        basename = '{}_{}_ranked_knobs'.format(dbms, hw).replace('.', '')
        with open(basename + '.txt', 'w') as f:
            f.write('\n'.join(knobs))

        django_entry = [{
            'model': MODEL,
            'fields': {
                'dbms': DBMSS[dbms],
                'hardware': HARDWARES[hw],
                'creation_timestamp': TIMESTAMP,
                'task_type': TASK_TYPE,
                'value': json.dumps(knobs, indent=4)
            }
        }]
        savepath = basename + '.json'
        with open(savepath, 'w') as f:
            json.dump(django_entry, f, indent=4)

        shutil.copy(savepath, '../../../preload/{}'.format(savepath))


if __name__ == '__main__':
    main()

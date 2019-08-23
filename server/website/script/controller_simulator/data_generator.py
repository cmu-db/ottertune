#
# OtterTune - data_generator.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Nov 30, 2017

@author: dvanaken
'''

import copy
import datetime
import logging
import os
import shutil
import sys

import json
import numpy as np

LOG = logging.getLogger(__name__)


# Data generator configuration
OBSERVATION_TIME_SEC = 300  # 5 minutes
START_TIME = datetime.datetime.now() - datetime.timedelta(weeks=1)
START_FREQUENCY = datetime.timedelta(minutes=10)
END_FREQUENCY = datetime.timedelta(seconds=OBSERVATION_TIME_SEC)
EPOCH = datetime.datetime.utcfromtimestamp(0)

# Paths
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SAMPLE_DIR = os.path.join(ROOT_DIR, 'samples')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'generated_data')


def unix_time_millis(dt):
    return int((dt - EPOCH).total_seconds() * 1000.0)


def generate_data(n_workloads, n_samples_per_workload):
    with open(os.path.join(SAMPLE_DIR, 'knobs.json'), 'r') as f:
        knob_sample = json.load(f)
    with open(os.path.join(SAMPLE_DIR, 'metrics_before.json'), 'r') as f:
        metrics_start_sample = json.load(f)
    with open(os.path.join(SAMPLE_DIR, 'metrics_after.json'), 'r') as f:
        metrics_end_sample = json.load(f)
    with open(os.path.join(SAMPLE_DIR, 'summary.json'), 'r') as f:
        summary_sample = json.load(f)

    start_time = START_TIME
    end_time = START_TIME + END_FREQUENCY

    for i in range(n_workloads):
        workload_name = 'workload-{}'.format(i)
        wkld_dir = os.path.join(OUTPUT_DIR, workload_name)
        os.mkdir(wkld_dir)

        for j in range(n_samples_per_workload):
            knob_data = copy.deepcopy(knob_sample)
            metrics_start_data = copy.deepcopy(metrics_start_sample)
            metrics_end_data = copy.deepcopy(metrics_end_sample)
            summary_data = copy.deepcopy(summary_sample)

            summary_data['workload_name'] = workload_name
            summary_data['observation_time'] = OBSERVATION_TIME_SEC
            summary_data['start_time'] = unix_time_millis(start_time)
            summary_data['end_time'] = unix_time_millis(end_time)
            start_time = start_time + START_FREQUENCY
            end_time = start_time + END_FREQUENCY

            knob_vals = np.random.randint(1, 11, 4)
            global_knobs = knob_data['global']['global']
            global_knobs['shared_buffers'] = str(knob_vals[0]) + 'GB'
            global_knobs['work_mem'] = str(knob_vals[1]) + 'GB'
            global_knobs['checkpoint_timing'] = str(knob_vals[2]) + 'min'
            global_knobs['effective_io_concurrency'] = str(knob_vals[3])

            metrics_start_data['global']['pg_stat_bgwriter']['buffers_alloc'] = np.random.randint(
                3000, 7000)
            metrics_end_data['global']['pg_stat_bgwriter']['buffers_alloc'] = np.random.randint(
                7000, 10000)

            locations = [
                ('xact_commit', metrics_start_data['local']['database']['pg_stat_database']),
                ('xact_commit', metrics_end_data['local']['database']['pg_stat_database']),
                ('n_tup_ins', metrics_start_data['local']['table']['pg_stat_user_tables']),
                ('n_tup_ins', metrics_end_data['local']['table']['pg_stat_user_tables']),
                ('idx_blks_hit', metrics_start_data['local']['indexes']['pg_statio_user_indexes']),
                ('idx_blks_hit', metrics_end_data['local']['indexes']['pg_statio_user_indexes']),
            ]

            for k, (name, loc) in enumerate(locations):

                for kvs in list(loc.values()):
                    if k % 2 == 0:  # start time must be smaller value
                        met_val = np.random.randint(30000, 70000)
                    else:
                        met_val = np.random.randint(70000, 100000)
                    kvs[name] = met_val

            basepath = os.path.join(wkld_dir, 'sample-{}'.format(j))

            with open(basepath + "__knobs.json", 'w') as f:
                json.dump(knob_data, f, indent=4)
            with open(basepath + '__metrics_start.json', 'w') as f:
                json.dump(metrics_start_data, f, indent=4)
            with open(basepath + '__metrics_end.json', 'w') as f:
                json.dump(metrics_end_data, f, indent=4)
            with open(basepath + '__summary.json', 'w') as f:
                json.dump(summary_data, f, indent=4)


def main():
    if len(sys.argv) < 3:
        LOG.error('Usage: python data_generator.py [n_workloads] [n_samples_per_workload] '
                  '[optional: random_seed]')
        sys.exit(1)
    if len(sys.argv) == 4:
        random_seed = int(sys.argv[3])
        LOG.info("Seeding the generator with value: %d", random_seed)
        np.random.seed(seed=random_seed)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.mkdir(OUTPUT_DIR)

    generate_data(int(sys.argv[1]), int(sys.argv[2]))
    LOG.info("Finished. Generated data written to %s.", OUTPUT_DIR)


if __name__ == "__main__":
    main()

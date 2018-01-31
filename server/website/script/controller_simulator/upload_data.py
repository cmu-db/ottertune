#
# OtterTune - upload_data.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Nov 30, 2017

@author: dvanaken
'''

import argparse
import glob
import logging
import os

import urllib2
from poster.encode import multipart_encode
from poster.streaminghttp import register_openers

register_openers()

# Logging
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)


def upload(basedir, upload_code):
    for wkld_dir in sorted(glob.glob(os.path.join(basedir, '*'))):
        LOG.info('Uploading sample for workload %s...', wkld_dir)
        sample_idx = 0
        while True:
            samples = glob.glob(os.path.join(wkld_dir, 'sample-{}__*').format(sample_idx))
            if len(samples) == 0:
                break
            assert len(samples) == 4
            basename = samples[0].split('__')[0]
            params = {
                'summary': open(basename + '__summary.json', "r"),
                'knobs': open(basename + '__knobs.json', "r"),
                'metrics_before': open(basename + '__metrics_start.json', 'r'),
                'metrics_after': open(basename + '__metrics_end.json', 'r'),
                'upload_code': upload_code,
            }

            datagen, headers = multipart_encode(params)
            request = urllib2.Request("http://0.0.0.0:8000/new_result/", datagen, headers)
            LOG.info("Response: %s\n", urllib2.urlopen(request).read())
            sample_idx += 1


def main():
    parser = argparse.ArgumentParser(description="Upload generated data to the website")
    parser.add_argument('datadir', type=str, nargs=1,
                        help='Directory containing the generated data')
    parser.add_argument('upload_code', type=str, nargs=1,
                        help='The website\'s upload code')
    args = parser.parse_args()
    upload(args.datadir, args.upload_code)


if __name__ == "__main__":
    main()

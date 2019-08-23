#
# OtterTune - upload_batch.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import argparse
import glob
import logging
import os
import requests


# Logging
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)


# Upload all files in the datadir to OtterTune's server side.
# You may want to upload your training data to the non-tuning session.
def upload_batch(datadir, upload_code, url):

    samples = glob.glob(os.path.join(datadir, '*__summary.json'))
    count = len(samples)
    samples_prefix = []

    LOG.info('Uploading %d samples in %s...', count, datadir)
    for sample in samples:
        prefix = sample.split('/')[-1].split('__')[0]
        samples_prefix.append(prefix)

    for i in range(count):
        prefix = samples_prefix[i]
        params = {
            'summary': open(os.path.join(datadir, '{}__summary.json'.format(prefix)), 'rb'),
            'knobs': open(os.path.join(datadir, '{}__knobs.json'.format(prefix)), 'rb'),
            'metrics_before': open(os.path.join(datadir,
                                                '{}__metrics_before.json'.format(prefix)), 'rb'),
            'metrics_after': open(os.path.join(datadir,
                                               '{}__metrics_after.json'.format(prefix)), 'rb'),
        }

        LOG.info('Upload %d-th sample %s__*.json', i + 1, prefix)
        response = requests.post(url,
                                 files=params,
                                 data={'upload_code': upload_code})
        LOG.info(response.content)


def main():
    parser = argparse.ArgumentParser(description="Upload generated data to the website")
    parser.add_argument('datadir', type=str, nargs=1,
                        help='Directory containing the generated data')
    parser.add_argument('upload_code', type=str, nargs=1,
                        help='The website\'s upload code')
    parser.add_argument('url', type=str, default='http://0.0.0.0:8000/new_result/',
                        nargs='?', help='The upload url: server_ip/new_result/')
    args = parser.parse_args()
    upload_batch(args.datadir[0], args.upload_code[0], args.url)


if __name__ == "__main__":
    main()

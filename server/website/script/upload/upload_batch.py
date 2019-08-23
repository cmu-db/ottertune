#
# OtterTune - upload_batch.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

import argparse
import logging
import os
import glob
import numpy as np
import requests

# Logging
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)


class ResultUploader(object):

    SUMMARY_EXT = '.summary'
    PARAMS_EXT = '.params'
    METRICS_EXT = '.metrics'
    SAMPLES_EXT = '.samples'
    EXPCFG_EXT = '.expconfig'
    RAW_EXT = '.csv'

    REQ_EXTS = [SUMMARY_EXT, PARAMS_EXT, METRICS_EXT, SAMPLES_EXT, EXPCFG_EXT]

    def __init__(self, upload_code, upload_url):
        self._upload_code = upload_code
        self._upload_url = upload_url

    def upload_batch(self, directories, max_files=5):
        for d in directories:
            cluster_name = os.path.basename(d)
            fnames = glob.glob(os.path.join(d, '*.summary'))
            if max_files < len(fnames):
                fnames = list(np.random.choice(fnames, max_files))
            bases = [fn.split('.summary')[0] for fn in fnames]

            # Verify required extensions exist
            for base in bases:
                complete = True
                for ext in self.REQ_EXTS:
                    next_file = base + ext
                    if not os.path.exists(next_file):
                        LOG.warning("WARNING: missing file %s, skipping...", next_file)
                        complete = False
                        break
                if not complete:
                    continue
                self.upload(base, cluster_name)

    def upload(self, basepath, cluster_name):
        exts = list(self.REQ_EXTS)
        if os.path.exists(basepath + self.RAW_EXT):
            exts.append(self.RAW_EXT)
        fhandlers = {ext: open(basepath + ext, 'rb') for ext in exts}
        params = {
            'summary_data': fhandlers[self.SUMMARY_EXT],
            'db_metrics_data': fhandlers[self.METRICS_EXT],
            'db_parameters_data': fhandlers[self.PARAMS_EXT],
            'sample_data': fhandlers[self.SAMPLES_EXT],
            'benchmark_conf_data': fhandlers[self.EXPCFG_EXT],
        }

        if self.RAW_EXT in fhandlers:
            params['raw_data'] = fhandlers[self.RAW_EXT]

        response = requests.post(self._upload_url,
                                 files=params,
                                 data={'upload_code': self._upload_code,
                                       'cluster_name': cluster_name})
        LOG.info(response.content)

        for fh in list(fhandlers.values()):
            fh.close()


def main():
    parser = argparse.ArgumentParser(description="Upload generated data to the website")
    parser.add_argument('upload_code', type=str, nargs=1,
                        help='The website\'s upload code')
    parser.add_argument('server', type=str, default='http://0.0.0.0:8000',
                        nargs='?', help='The server\'s address (ip:port)')
    args = parser.parse_args()
    url = args.server + '/new_result/'
    upload_code = args.upload_code[0]
    uploader = ResultUploader(upload_code, url)
    dirnames = glob.glob(os.path.join(os.path.expanduser(
        '~'), 'Dropbox/Apps/ottertune/data/sample_data/exps_*'))[:2]
    uploader.upload_batch(dirnames, max_files=3)


if __name__ == '__main__':
    main()

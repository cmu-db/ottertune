#
# OtterTune - upload.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import argparse
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


def upload(datadir, upload_code, url):
    params = {
        'summary': open(os.path.join(datadir, 'summary.json'), "r"),
        'knobs': open(os.path.join(datadir, 'knobs.json'), "r"),
        'metrics_before': open(os.path.join(datadir, 'metrics_before.json'), 'r'),
        'metrics_after': open(os.path.join(datadir, 'metrics_after.json'), 'r'),
        'upload_code': upload_code,
    }
    datagen, headers = multipart_encode(params)
    request = urllib2.Request(url, datagen, headers)
    LOG.info(urllib2.urlopen(request).read())


def main():
    parser = argparse.ArgumentParser(description="Upload generated data to the website")
    parser.add_argument('datadir', type=str, nargs=1,
                        help='Directory containing the generated data')
    parser.add_argument('upload_code', type=str, nargs=1,
                        help='The website\'s upload code')
    parser.add_argument('url', type=str, default='http://0.0.0.0:8000/new_result/',
                        nargs='?', help='The upload url: server_ip/new_result/')
    args = parser.parse_args()
    upload(args.datadir[0], args.upload_code[0], args.url)


if __name__ == "__main__":
    main()

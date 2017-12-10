'''
Created on Nov 30, 2017

@author: dvanaken
'''

import glob
import os
import sys

from poster.encode import multipart_encode
from poster.streaminghttp import register_openers
import urllib2

register_openers()

def upload(basedir, upload_code):
    for wkld_dir in sorted(glob.glob(os.path.join(basedir, '*'))):
        print wkld_dir
        sample_idx = 0
        while True:
            samples = glob.glob(os.path.join(wkld_dir, 'sample-{}__*').format(sample_idx))
            if len(samples) == 0:
                break
            assert len(samples) == 4
            basename = samples[0].split('__')[0]
            params = {
                'summary': open(basename + '__summary.json', "r"),
                'knobs': open(basename + '__knobs.json',"r"),
                'metrics_before':open(basename + '__metrics_start.json','r'),
                'metrics_after':open(basename + '__metrics_end.json','r'),
                'upload_code':  upload_code,
            }

            datagen, headers = multipart_encode(params)
            request = urllib2.Request("http://0.0.0.0:8000/new_result/", datagen, headers)
            print urllib2.urlopen(request).read()
            sample_idx += 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python upload_data.py [datadir] [upload_code]"
        sys.exit(1)
    upload(sys.argv[1], sys.argv[2])

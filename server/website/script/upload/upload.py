import os
import sys
from poster.encode import multipart_encode
from poster.streaminghttp import register_openers
import urllib2

register_openers()

def upload(upload_code, datadir):
    params = {
        'summary': open(os.path.join(datadir, 'sample-0__summary.json'), "r"),
        'knobs': open(os.path.join(datadir, 'sample-0__knobs.json'),"r"),
        'metrics_start':open(os.path.join(datadir, 'sample-0__metrics_start.json'),'r'),
        'metrics_end':open(os.path.join(datadir, 'sample-0__metrics_end.json'),'r'),
        'upload_code':  upload_code,
    }
    
    datagen, headers = multipart_encode(params)
    
    request = urllib2.Request("http://0.0.0.0:8000/new_result/", datagen, headers)

    print urllib2.urlopen(request).read()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python upload.py [upload_code] [path_to_sample_data]"
    upload(sys.argv[1], sys.argv[2])

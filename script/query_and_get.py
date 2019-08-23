#
# OtterTune - query_and_get.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Feb 11, 2018

@author: taodai
'''

import sys
import time
import logging
import json
import urllib.request

# Logging
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)


# take 3 arguments, save result to next_config in working directory
# base_url: for instance, https://0.0.0.0:8000/
# upload_code: upload code...
# query_interval: time (in second) between queries
def main():
    base_url = sys.argv[1].strip('/')
    upload_code = sys.argv[2]
    query_interval = int(sys.argv[3])
    request = base_url + '/query_and_get/' + upload_code
    timer = 0
    start = time.time()
    while True:
        response = urllib.request.urlopen(request).read().decode()
        if 'Fail' in response:
            LOG.info('Tuning failed\n')
            break
        elif response == 'null' or 'not ready' in response:
            time.sleep(query_interval)
            timer += query_interval
            LOG.info('%s s\n', str(timer))
        else:
            next_conf_f = open('next_config', 'w')
            next_conf_f.write(json.loads(response))
            next_conf_f.close()
            break
    elapsed_time = time.time() - start
    LOG.info('Elapsed time: %s\n', str(elapsed_time))


if __name__ == "__main__":
    main()

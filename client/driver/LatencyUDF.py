#
# OtterTune - LatencyUDF.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

import sys
import json
from collections import OrderedDict


def main():
    if (len(sys.argv) != 2):
        raise Exception("Usage: python udf.py [Output Directory]")

    with open(sys.argv[1] + "/summary.json", "r") as f:
        conf = json.load(f,
                         encoding="UTF-8",
                         object_pairs_hook=OrderedDict)
        start_time = conf['start_time']
        end_time = conf['end_time']

    with open(sys.argv[1] + "/metrics_before.json", "r") as f:
        conf_before = json.load(f,
                                encoding="UTF-8",
                                object_pairs_hook=OrderedDict)
        conf_before['global']['udf'] = OrderedDict([("latency", "0")])

    with open(sys.argv[1] + "/metrics_after.json", "r") as f:
        conf_after = json.load(f,
                               encoding="UTF-8",
                               object_pairs_hook=OrderedDict)
        conf_after['global']['udf'] = OrderedDict([("latency", str(end_time - start_time))])

    with open(sys.argv[1] + "/metrics_before.json", "w") as f:
        f.write(json.dumps(conf_before, indent=4))

    with open(sys.argv[1] + "/metrics_after.json", "w") as f:
        f.write(json.dumps(conf_after, indent=4))


if __name__ == "__main__":
    main()

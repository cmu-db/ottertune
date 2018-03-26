#
# OtterTune - PostgresConf.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Mar 23, 2018
@author: Jacky, bohan
'''

import sys
import json
from collections import OrderedDict


def main():
    if (len(sys.argv) != 3):
        raise Exception("Usage: python confparser.py [Next Config] [Current Config]")

    with open(sys.argv[1], "r") as f:
        conf = json.load(f,
                         encoding="UTF-8",
                         object_pairs_hook=OrderedDict)
    conf = conf['recommendation']
    with open(sys.argv[2], "r+") as postgresqlconf:
        lines = postgresqlconf.readlines()
        settings_idx = lines.index("# Add settings for extensions here\n")
        postgresqlconf.seek(0)
        postgresqlconf.truncate(0)

        lines = lines[0:(settings_idx + 1)]
        for line in lines:
            postgresqlconf.write(line)

        for (knob_name, knob_value) in conf.iteritems():
            postgresqlconf.write(str(knob_name) + " = " + str(knob_value) + "\n")


if __name__ == "__main__":
    main()

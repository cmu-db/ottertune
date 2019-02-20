#
# OtterTune - OracleConf.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Feb 20, 2019
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
    with open(sys.argv[2], "r+") as oracleconf:
        lines = oracleconf.readlines()
	  
        signal = "# configurations recommended by ottertune:\n"
        settings_idx = lines.find(signal)
        if settings_idx == -1:
            oracleconf.write(signal)
            oracleconf.flush();
        settings_idx = lines.find(signal)

        postgresqlconf.seek(0)
        postgresqlconf.truncate(0)

        lines = lines[0:(settings_idx + 1)]
        for line in lines:
            postgresqlconf.write(line)

        for (knob_name, knob_value) in list(conf.items()):
            postgresqlconf.write(str(knob_name) + " = " + str(knob_value) + "\n")


if __name__ == "__main__":
    main()

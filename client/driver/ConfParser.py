#
# OtterTune - ConfParser.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Mar 23, 2018
@author: Jacky, bohan, Dongsheng
'''

import sys
import json
from collections import OrderedDict


def change_postgres_conf(recommendation, postgresqlconf):
    lines = postgresqlconf.readlines()
    settings_idx = lines.index("# Add settings for extensions here\n")
    postgresqlconf.seek(0)
    postgresqlconf.truncate(0)

    lines = lines[0:(settings_idx + 1)]
    for line in lines:
        postgresqlconf.write(line)

    for (knob_name, knob_value) in list(recommendation.items()):
        postgresqlconf.write(str(knob_name) + " = " + str(knob_value) + "\n")


def change_oracle_conf(recommendation, oracle_conf):
    lines = oracle_conf.readlines()
    signal = "# configurations recommended by ottertune:\n"
    if signal not in lines:
        oracle_conf.write('\n' + signal)
        oracle_conf.flush()
        oracle_conf.seek(0)
        lines = oracle_conf.readlines()
    settings_idx = lines.index(signal)

    oracle_conf.seek(0)
    oracle_conf.truncate(0)

    lines = lines[0:(settings_idx + 1)]
    for line in lines:
        oracle_conf.write(line)

    for (knob_name, knob_value) in list(recommendation.items()):
        oracle_conf.write(str(knob_name) + " = " + str(knob_value).strip('B') + "\n")


def main():
    if len(sys.argv) != 4:
        raise Exception("Usage: python [DB type] ConfParser.py [Next Config] [Current Config]")
    database_type = sys.argv[1]
    next_config_name = sys.argv[2]
    cur_config_name = sys.argv[3]
    with open(next_config_name, 'r') as next_config, open(cur_config_name, 'r+') as cur_config:
        config = json.load(next_config, encoding="UTF-8", object_pairs_hook=OrderedDict)
        recommendation = config['recommendation']
        if database_type == 'postgres':
            change_postgres_conf(recommendation, cur_config)
        elif database_type == 'oracle':
            change_oracle_conf(recommendation, cur_config)
        else:
            raise Exception("Database Type {} Not Implemented !".format(database_type))


if __name__ == "__main__":
    main()

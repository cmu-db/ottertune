#
# OtterTune - fabfile.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Mar 23, 2018

@author: bohan
'''
import sys
import json
import logging
import time
import os.path
import re
import glob
from multiprocessing import Process
from fabric.api import (env, local, task, lcd)
from fabric.state import output as fabric_output

LOG = logging.getLogger()
LOG.setLevel(logging.DEBUG)
Formatter = logging.Formatter(  # pylint: disable=invalid-name
    "%(asctime)s [%(levelname)s]  %(message)s")

# print the log
ConsoleHandler = logging.StreamHandler(sys.stdout)  # pylint: disable=invalid-name
ConsoleHandler.setFormatter(Formatter)
LOG.addHandler(ConsoleHandler)

# Fabric environment settings
env.hosts = ['localhost']
fabric_output.update({
    'running': True,
    'stdout': True,
})

# intervals of restoring the databse
RELOAD_INTERVAL = 10
# maximum disk usage
MAX_DISK_USAGE = 90

with open('driver_config.json', 'r') as f:
    CONF = json.load(f)


@task
def check_disk_usage():
    partition = CONF['database_disk']
    disk_use = 0
    cmd = "df -h {}".format(partition)
    out = local(cmd, capture=True).splitlines()[1]
    m = re.search('\d+(?=%)', out)  # pylint: disable=anomalous-backslash-in-string
    if m:
        disk_use = int(m.group(0))
    LOG.info("Current Disk Usage: %s%s", disk_use, '%')
    return disk_use


@task
def check_memory_usage():
    cmd = 'free -m -h'
    local(cmd)


@task
def restart_database():
    if CONF['database_type'] == 'postgres':
        cmd = 'sudo service postgresql restart'
    elif CONF['database_type'] == 'oracle':
        cmd = 'sh oracleScripts/shutdownOracle.sh && sh oracleScripts/startupOracle.sh'
    else:
        raise Exception("Database Type {} Not Implemented !".format(CONF['database_type']))
    local(cmd)


@task
def drop_database():
    if CONF['database_type'] == 'postgres':
        cmd = "PGPASSWORD={} dropdb -e --if-exists {} -U {}".\
              format(CONF['password'], CONF['database_name'], CONF['username'])
    else:
        raise Exception("Database Type {} Not Implemented !".format(CONF['database_type']))
    local(cmd)


@task
def create_database():
    if CONF['database_type'] == 'postgres':
        cmd = "PGPASSWORD={} createdb -e {} -U {}".\
              format(CONF['password'], CONF['database_name'], CONF['username'])
    else:
        raise Exception("Database Type {} Not Implemented !".format(CONF['database_type']))
    local(cmd)


@task
def change_conf():
    next_conf = 'next_config'
    cmd = "sudo python3 ConfParser.py {} {} {}".\
          format(CONF['database_type'], next_conf, CONF['database_conf'])
    local(cmd)


@task
def load_oltpbench():
    cmd = "./oltpbenchmark -b {} -c {} --create=true --load=true".\
          format(CONF['oltpbench_workload'], CONF['oltpbench_config'])
    with lcd(CONF['oltpbench_home']):  # pylint: disable=not-context-manager
        local(cmd)


@task
def run_oltpbench():
    cmd = "./oltpbenchmark -b {} -c {} --execute=true -s 5 -o outputfile".\
          format(CONF['oltpbench_workload'], CONF['oltpbench_config'])
    with lcd(CONF['oltpbench_home']):  # pylint: disable=not-context-manager
        local(cmd)


@task
def run_oltpbench_bg():
    cmd = "./oltpbenchmark -b {} -c {} --execute=true -s 5 -o outputfile > {} 2>&1 &".\
          format(CONF['oltpbench_workload'], CONF['oltpbench_config'], CONF['oltpbench_log'])
    with lcd(CONF['oltpbench_home']):  # pylint: disable=not-context-manager
        local(cmd)


@task
def run_controller():
    cmd = 'sudo gradle run -PappArgs="-c {} -d output/" --no-daemon > {}'.\
          format(CONF['controller_config'], CONF['controller_log'])
    with lcd("../controller"):  # pylint: disable=not-context-manager
        local(cmd)


@task
def signal_controller():
    pid = int(open('../controller/pid.txt').read())
    cmd = 'sudo kill -2 {}'.format(pid)
    with lcd("../controller"):  # pylint: disable=not-context-manager
        local(cmd)


@task
def save_dbms_result():
    t = int(time.time())
    files = ['knobs.json', 'metrics_after.json', 'metrics_before.json', 'summary.json']
    for f_ in files:
        f_prefix = f_.split('.')[0]
        cmd = 'cp ../controller/output/{} {}/{}__{}.json'.\
              format(f_, CONF['save_path'], t, f_prefix)
        local(cmd)


@task
def free_cache():
    cmd = 'sync; sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"'
    local(cmd)


@task
def upload_result():
    cmd = 'python3 ../../server/website/script/upload/upload.py \
           ../controller/output/ {} {}/new_result/'.format(CONF['upload_code'],
                                                           CONF['upload_url'])
    local(cmd)


@task
def get_result():
    cmd = 'python3 ../../script/query_and_get.py {} {} 5'.\
          format(CONF['upload_url'], CONF['upload_code'])
    local(cmd)


@task
def add_udf():
    cmd = 'sudo python3 ./LatencyUDF.py ../controller/output/'
    local(cmd)


@task
def upload_batch():
    cmd = 'python3 ./upload_batch.py {} {} {}/new_result/'.format(CONF['save_path'],
                                                                  CONF['upload_code'],
                                                                  CONF['upload_url'])
    local(cmd)


@task
def dump_database():
    db_file_path = '{}/{}.dump'.format(CONF['database_save_path'], CONF['database_name'])
    if os.path.exists(db_file_path):
        LOG.info('%s already exists ! ', db_file_path)
        return False
    else:
        LOG.info('Dump database %s to %s', CONF['database_name'], db_file_path)
        # You must create a directory named dpdata through sqlplus in your Oracle database
        if CONF['database_type'] == 'oracle':
            cmd = 'expdp {}/{}@{} schemas={} dumpfile={}.dump DIRECTORY=dpdata'.format(
                'c##tpcc', 'oracle', 'orcldb', 'c##tpcc', 'orcldb')
        elif CONF['database_type'] == 'postgres':
            cmd = 'PGPASSWORD={} pg_dump -U {} -F c -d {} > {}'.format(CONF['password'],
                                                                       CONF['username'],
                                                                       CONF['database_name'],
                                                                       db_file_path)
        else:
            raise Exception("Database Type {} Not Implemented !".format(CONF['database_type']))
        local(cmd)
        return True


@task
def restore_database():
    if CONF['database_type'] == 'oracle':
        # You must create a directory named dpdata through sqlplus in your Oracle database
        # The following script assumes such directory exists.
        # You may want to modify the username, password, and dump file name in the script
        cmd = 'sh oracleScripts/restoreOracle.sh'
    elif CONF['database_type'] == 'postgres':
        db_file_path = '{}/{}.dump'.format(CONF['database_save_path'], CONF['database_name'])
        drop_database()
        create_database()
        cmd = 'PGPASSWORD={} pg_restore -U {} -n public -j 8 -F c -d {} {}'.\
              format(CONF['password'], CONF['username'], CONF['database_name'], db_file_path)
    else:
        raise Exception("Database Type {} Not Implemented !".format(CONF['database_type']))
    LOG.info('Start restoring database')
    local(cmd)
    LOG.info('Finish restoring database')


def _ready_to_start_oltpbench():
    return (os.path.exists(CONF['controller_log']) and
            'Output the process pid to'
            in open(CONF['controller_log']).read())


def _ready_to_start_controller():
    return (os.path.exists(CONF['oltpbench_log']) and
            'Warmup complete, starting measurements'
            in open(CONF['oltpbench_log']).read())


def _ready_to_shut_down_controller():
    pid_file_path = '../controller/pid.txt'
    return (os.path.exists(pid_file_path) and os.path.exists(CONF['oltpbench_log']) and
            'Output throughput samples into file' in open(CONF['oltpbench_log']).read())


def clean_logs():
    # remove oltpbench log
    cmd = 'rm -f {}'.format(CONF['oltpbench_log'])
    local(cmd)

    # remove controller log
    cmd = 'rm -f {}'.format(CONF['controller_log'])
    local(cmd)


@task
def lhs_samples(count=10):
    cmd = 'python3 lhs.py {} {} {}'.format(count, CONF['lhs_knob_path'], CONF['lhs_save_path'])
    local(cmd)


@task
def loop():

    # free cache
    free_cache()

    # remove oltpbench log and controller log
    clean_logs()

    # restart database
    restart_database()

    # check disk usage
    if check_disk_usage() > MAX_DISK_USAGE:
        LOG.WARN('Exceeds max disk usage %s', MAX_DISK_USAGE)

    # run controller from another process
    p = Process(target=run_controller, args=())
    p.start()
    LOG.info('Run the controller')

    # run oltpbench as a background job
    while not _ready_to_start_oltpbench():
        pass
    run_oltpbench_bg()
    LOG.info('Run OLTP-Bench')

    # the controller starts the first collection
    while not _ready_to_start_controller():
        pass
    signal_controller()
    LOG.info('Start the first collection')

    # stop the experiment
    while not _ready_to_shut_down_controller():
        pass
    signal_controller()
    LOG.info('Start the second collection, shut down the controller')

    p.join()

    # add user defined target objective
    # add_udf()

    # save result
    save_dbms_result()

    # upload result
    upload_result()

    # get result
    get_result()

    # change config
    change_conf()


@task
def run_lhs():
    datadir = CONF['lhs_save_path']
    samples = glob.glob(os.path.join(datadir, 'config_*'))

    # dump database if it's not done before.
    dump = dump_database()

    for i, sample in enumerate(samples):
        # reload database periodically
        if RELOAD_INTERVAL > 0:
            if i % RELOAD_INTERVAL == 0:
                if i == 0 and dump is False:
                    restore_database()
                elif i > 0:
                    restore_database()
        # free cache
        free_cache()

        LOG.info('\n\n Start %s-th sample %s \n\n', i, sample)
        # check memory usage
        # check_memory_usage()

        # check disk usage
        if check_disk_usage() > MAX_DISK_USAGE:
            LOG.WARN('Exceeds max disk usage %s', MAX_DISK_USAGE)

        # copy lhs-sampled config to the to-be-used config
        cmd = 'cp {} next_config'.format(sample)
        local(cmd)

        # remove oltpbench log and controller log
        clean_logs()

        # change config
        change_conf()

        # restart database
        restart_database()

        if CONF.get('oracle_awr_enabled', False):
            # create snapshot for oracle AWR report
            if CONF['database_type'] == 'oracle':
                local('sh snapshotOracle.sh')

        # run controller from another process
        p = Process(target=run_controller, args=())
        p.start()

        # run oltpbench as a background job
        while not _ready_to_start_oltpbench():
            pass
        run_oltpbench_bg()
        LOG.info('Run OLTP-Bench')

        while not _ready_to_start_controller():
            pass
        signal_controller()
        LOG.info('Start the first collection')

        while not _ready_to_shut_down_controller():
            pass
        # stop the experiment
        signal_controller()
        LOG.info('Start the second collection, shut down the controller')

        p.join()

        # save result
        save_dbms_result()

        # upload result
        upload_result()

        if CONF.get('oracle_awr_enabled', False):
            # create oracle AWR report for performance analysis
            if CONF['database_type'] == 'oracle':
                local('sh oracleScripts/snapshotOracle.sh && sh oracleScripts/awrOracle.sh')


@task
def run_loops(max_iter=1):
    # dump database if it's not done before.
    dump = dump_database()

    for i in range(int(max_iter)):
        if RELOAD_INTERVAL > 0:
            if i % RELOAD_INTERVAL == 0:
                if i == 0 and dump is False:
                    restore_database()
                elif i > 0:
                    restore_database()

        LOG.info('The %s-th Loop Starts / Total Loops %s', i + 1, max_iter)
        loop()
        LOG.info('The %s-th Loop Ends / Total Loops %s', i + 1, max_iter)

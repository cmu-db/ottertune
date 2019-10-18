#
# OtterTune - fabfile.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Mar 23, 2018

@author: bohan
'''
import glob
import json
import os
import re
import time
from collections import OrderedDict
from multiprocessing import Process

import logging
from logging.handlers import RotatingFileHandler

import requests
from fabric.api import env, local, task, lcd
from fabric.state import output as fabric_output

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
# Postgres datadir
PG_DATADIR = '/var/lib/postgresql/9.6/main'

# Load config
with open('driver_config.json', 'r') as _f:
    CONF = {k: os.path.expanduser(v) if isinstance(v, str) and v.startswith('~') else v
            for k, v in json.load(_f).items()}

# Create output directories
for _dir in (CONF['database_save_path'], CONF['log_path'], CONF['save_path'],
             CONF['lhs_save_path']):
    os.makedirs(_dir, exist_ok=True)

# Define paths
CONF['driver_log'] = os.path.join(CONF['log_path'], 'driver.log')
CONF['oltpbench_log'] = os.path.join(CONF['log_path'], 'oltpbench.log')
CONF['controller_log'] = os.path.join(CONF['log_path'], 'controller.log')
CONF['controller_config'] = os.path.join(CONF['controller_home'], 'config',
                                         '{}_config.json'.format(CONF['database_type']))

# Configure logging
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)
Formatter = logging.Formatter(  # pylint: disable=invalid-name
    fmt='%(asctime)s [%(funcName)s:%(lineno)03d] %(levelname)-5s: %(message)s',
    datefmt='%m-%d-%Y %H:%M:%S')
ConsoleHandler = logging.StreamHandler()  # pylint: disable=invalid-name
ConsoleHandler.setFormatter(Formatter)
LOG.addHandler(ConsoleHandler)
FileHandler = RotatingFileHandler(  # pylint: disable=invalid-name
    CONF['driver_log'], maxBytes=50000, backupCount=2)
FileHandler.setFormatter(Formatter)
LOG.addHandler(FileHandler)


def _parse_bool(value):
    if not isinstance(value, bool):
        value = str(value).lower() == 'true'
    return value


@task
def check_disk_usage():
    partition = CONF['database_disk']
    disk_use = 0
    if partition:
        cmd = "df -h {}".format(partition)
        out = local(cmd, capture=True).splitlines()[1]
        m = re.search(r'\d+(?=%)', out)
        if m:
            disk_use = int(m.group(0))
        LOG.info("Current Disk Usage: %s%s", disk_use, '%')
    return disk_use


@task
def check_memory_usage():
    cmd = 'free -m -h'
    local(cmd)


@task
def create_controller_config():
    if CONF['database_type'] == 'postgres':
        dburl_fmt = 'jdbc:postgresql://localhost:5432/{db}'.format
    elif CONF['database_type'] == 'oracle':
        dburl_fmt = 'jdbc:oracle:thin:@localhost:1521:{db}'.format
    else:
        raise Exception("Database Type {} Not Implemented !".format(CONF['database_type']))

    config = dict(
        database_type=CONF['database_type'],
        database_url=dburl_fmt(db=CONF['database_name']),
        username=CONF['username'],
        password=CONF['password'],
        upload_code='DEPRECATED',
        upload_url='DEPRECATED',
        workload_name=CONF['oltpbench_workload']
    )

    with open(CONF['controller_config'], 'w') as f:
        json.dump(config, f, indent=2)


@task
def restart_database():
    if CONF['database_type'] == 'postgres':
        cmd = 'sudo -u postgres pg_ctl -D {} -w restart'.format(PG_DATADIR)
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
def reset_conf():
    change_conf()


@task
def change_conf(next_conf=None):
    signal = "# configurations recommended by ottertune:\n"
    next_conf = next_conf or {}

    with open(CONF['database_conf'], 'r') as f:
        lines = f.readlines()

    if signal not in lines:
        lines += ['\n', signal]

    signal_idx = lines.index(signal)
    lines = lines[0:signal_idx + 1]

    if isinstance(next_conf, str):
        with open(next_conf, 'r') as f:
            recommendation = json.load(
                f, encoding="UTF-8", object_pairs_hook=OrderedDict)['recommendation']
    else:
        recommendation = next_conf

    assert isinstance(recommendation, dict)

    for name, value in recommendation.items():
        if CONF['database_type'] == 'oracle' and isinstance(value, str):
            value = value.strip('B')
        lines.append('{} = {}\n'.format(name, value))
    lines.append('\n')

    tmpconf = 'tmp_' + os.path.basename(CONF['database_conf'])
    with open(tmpconf, 'w') as f:
        f.write(''.join(lines))

    local('sudo cp {0} {0}.ottertune.bak'.format(CONF['database_conf']))
    local('sudo mv {} {}'.format(tmpconf, CONF['database_conf']))


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
    if not os.path.exists(CONF['controller_config']):
        create_controller_config()
    cmd = 'gradle run -PappArgs="-c {} -d output/" --no-daemon > {}'.\
          format(CONF['controller_config'], CONF['controller_log'])
    with lcd(CONF['controller_home']):  # pylint: disable=not-context-manager
        local(cmd)


@task
def signal_controller():
    pidfile = os.path.join(CONF['controller_home'], 'pid.txt')
    with open(pidfile, 'r') as f:
        pid = int(f.read())
    cmd = 'sudo kill -2 {}'.format(pid)
    with lcd(CONF['controller_home']):  # pylint: disable=not-context-manager
        local(cmd)


@task
def save_dbms_result():
    t = int(time.time())
    files = ['knobs.json', 'metrics_after.json', 'metrics_before.json', 'summary.json']
    for f_ in files:
        srcfile = os.path.join(CONF['controller_home'], 'output', f_)
        dstfile = os.path.join(CONF['save_path'], '{}__{}'.format(t, f_))
        local('cp {} {}'.format(srcfile, dstfile))
    return t


@task
def save_next_config(next_config, t=None):
    if not t:
        t = int(time.time())
    with open(os.path.join(CONF['save_path'], '{}__next_config.json'.format(t)), 'w') as f:
        json.dump(next_config, f, indent=2)
    return t


@task
def free_cache():
    cmd = 'sync; sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"'
    local(cmd)


@task
def upload_result(result_dir=None, prefix=None):
    result_dir = result_dir or os.path.join(CONF['controller_home'], 'output')
    prefix = prefix or ''

    files = {}
    for base in ('summary', 'knobs', 'metrics_before', 'metrics_after'):
        fpath = os.path.join(result_dir, prefix + base + '.json')

        # Replaces the true db version with the specified version to allow for
        # testing versions not officially supported by OtterTune
        if base == 'summary' and 'override_database_version' in CONF and \
                CONF['override_database_version']:
            with open(fpath, 'r') as f:
                summary = json.load(f)
            summary['real_database_version'] = summary['database_version']
            summary['database_version'] = CONF['override_database_version']
            with open(fpath, 'w') as f:
                json.dump(summary, f, indent=1)

        files[base] = open(fpath, 'rb')

    response = requests.post(CONF['upload_url'] + '/new_result/', files=files,
                             data={'upload_code': CONF['upload_code']})
    if response.status_code != 200:
        raise Exception('Error uploading result.\nStatus: {}\nMessage: {}\n'.format(
            response.status_code, response.content))

    for f in files.values():  # pylint: disable=not-an-iterable
        f.close()

    LOG.info(response.content)

    return response


@task
def get_result(max_time_sec=180, interval_sec=5):
    max_time_sec = int(max_time_sec)
    interval_sec = int(interval_sec)
    url = CONF['upload_url'] + '/query_and_get/' + CONF['upload_code']
    elapsed = 0
    response_dict = None
    response = ''

    while elapsed <= max_time_sec:
        rsp = requests.get(url)
        response = rsp.content.decode()
        assert response != 'null'

        LOG.debug('%s [status code: %d, content_type: %s, elapsed: %ds]', response,
                  rsp.status_code, rsp.headers.get('content-type', ''), elapsed)

        if rsp.status_code == 200:
            # Success
            response_dict = json.loads(rsp.json(), object_pairs_hook=OrderedDict)
            break

        elif rsp.status_code == 202:
            # Not ready
            time.sleep(interval_sec)
            elapsed += interval_sec

        elif rsp.status_code == 400:
            # Failure
            raise Exception(
                "Failed to download the next config.\nStatus code: {}\nMessage: {}\n".format(
                    rsp.status_code, response))

        else:
            raise NotImplementedError(
                "Unhandled status code: '{}'.\nMessage: {}".format(rsp.status_code, response))

    if not response_dict:
        assert elapsed > max_time_sec, \
            'response={} but elapsed={}s <= max_time={}s'.format(
                response, elapsed, max_time_sec)
        raise Exception(
            'Failed to download the next config in {}s: {} (elapsed: {}s)'.format(
                max_time_sec, response, elapsed))

    LOG.info('Downloaded the next config in %ds: %s', elapsed, json.dumps(response_dict, indent=4))

    return response_dict


@task
def download_debug_info(pprint=False):
    pprint = _parse_bool(pprint)
    url = '{}/dump/{}'.format(CONF['upload_url'], CONF['upload_code'])
    params = {'pp': int(True)} if pprint else {}
    rsp = requests.get(url, params=params)

    if rsp.status_code != 200:
        raise Exception('Error downloading debug info.')

    filename = rsp.headers.get('Content-Disposition').split('=')[-1]
    file_len, exp_len = len(rsp.content), int(rsp.headers.get('Content-Length'))
    assert file_len == exp_len, 'File {}: content length != expected length: {} != {}'.format(
        filename, file_len, exp_len)

    with open(filename, 'wb') as f:
        f.write(rsp.content)
    LOG.info('Downloaded debug info to %s', filename)

    return filename


@task
def add_udf():
    cmd = 'sudo python3 ./LatencyUDF.py ../controller/output/'
    local(cmd)


@task
def upload_batch(result_dir=None, sort=True):
    result_dir = result_dir or CONF['save_path']
    sort = _parse_bool(sort)
    results = glob.glob(os.path.join(result_dir, '*__summary.json'))
    if sort:
        results = sorted(results)
    count = len(results)

    LOG.info('Uploading %d samples from %s...', count, result_dir)
    for i, result in enumerate(results):
        prefix = os.path.basename(result)
        prefix_len = os.path.basename(result).find('_') + 2
        prefix = prefix[:prefix_len]
        upload_result(result_dir=result_dir, prefix=prefix)
        LOG.info('Uploaded result %d/%d: %s__*.json', i + 1, count, prefix)


@task
def dump_database():
    db_file_path = os.path.join(CONF['database_save_path'], CONF['database_name'] + '.dump')
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
    ready = False
    if os.path.exists(CONF['controller_log']):
        with open(CONF['controller_log'], 'r') as f:
            content = f.read()
        ready = 'Output the process pid to' in content
    return ready


def _ready_to_start_controller():
    ready = False
    if os.path.exists(CONF['oltpbench_log']):
        with open(CONF['oltpbench_log'], 'r') as f:
            content = f.read()
        ready = 'Warmup complete, starting measurements' in content
    return ready


def _ready_to_shut_down_controller():
    pidfile = os.path.join(CONF['controller_home'], 'pid.txt')
    ready = False
    if os.path.exists(pidfile) and os.path.exists(CONF['oltpbench_log']):
        with open(CONF['oltpbench_log'], 'r') as f:
            content = f.read()
        ready = 'Output throughput samples into file' in content
    return ready


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
        LOG.warning('Exceeds max disk usage %s', MAX_DISK_USAGE)

    # run controller from another process
    p = Process(target=run_controller, args=())
    p.start()
    LOG.info('Run the controller')

    # run oltpbench as a background job
    while not _ready_to_start_oltpbench():
        time.sleep(1)
    run_oltpbench_bg()
    LOG.info('Run OLTP-Bench')

    # the controller starts the first collection
    while not _ready_to_start_controller():
        time.sleep(1)
    signal_controller()
    LOG.info('Start the first collection')

    # stop the experiment
    while not _ready_to_shut_down_controller():
        time.sleep(1)
    signal_controller()
    LOG.info('Start the second collection, shut down the controller')

    p.join()

    # add user defined target objective
    # add_udf()

    # save result
    result_timestamp = save_dbms_result()

    # upload result
    upload_result()

    # get result
    response = get_result()

    # save next config
    save_next_config(response, t=result_timestamp)

    # change config
    change_conf(response['recommendation'])


@task
def run_lhs():
    datadir = CONF['lhs_save_path']
    samples = glob.glob(os.path.join(datadir, 'config_*'))

    # dump database if it's not done before.
    dump = dump_database()

    result_timestamp = None
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
            LOG.warning('Exceeds max disk usage %s', MAX_DISK_USAGE)

        # load the next lhs-sampled config
        with open(sample, 'r') as f:
            next_config = json.load(f, object_pairs_hook=OrderedDict)
        save_next_config(next_config, t=result_timestamp)

        # remove oltpbench log and controller log
        clean_logs()

        # change config
        change_conf(next_config)

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
        result_timestamp = save_dbms_result()

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

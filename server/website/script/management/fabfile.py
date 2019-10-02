#
# OtterTune - fabfile.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Admin tasks

@author: dvanaken
'''

import logging
from collections import namedtuple
from fabric.api import env, local, quiet, settings, task
from fabric.state import output as fabric_output

from website.settings import DATABASES, PROJECT_ROOT

LOG = logging.getLogger(__name__)


# Fabric environment settings
env.hosts = ['localhost']
fabric_output.update({
    'running': False,
    'stdout': True,
})

Status = namedtuple('Status', ['RUNNING', 'STOPPED'])
STATUS = Status(0, 1)


# Setup and base commands
RABBITMQ_CMD = 'sudo rabbitmqctl {action}'.format


@task
def start_rabbitmq(detached=True):
    detached = parse_bool(detached)
    cmd = 'sudo rabbitmq-server' + (' -detached' if detached else '')
    local(cmd)


@task
def stop_rabbitmq():
    with settings(warn_only=True):
        local(RABBITMQ_CMD(action='stop'))


@task
def status_rabbitmq():
    with settings(warn_only=True), quiet():
        res = local(RABBITMQ_CMD(action='status'), capture=True)
    if res.return_code == 2 or res.return_code == 69:
        status = STATUS.STOPPED
    elif res.return_code == 0:
        status = STATUS.RUNNING
    else:
        raise Exception("Rabbitmq: unknown status " + str(res.return_code))
    log_status(status, 'rabbitmq')
    return status


@task
def start_celery():
    if status_rabbitmq() == STATUS.STOPPED:
        start_rabbitmq()
    local('python manage.py celery worker --detach --loglevel=info --pool=threads')


@task
def stop_celery():
    with settings(warn_only=True), quiet():
        local('kill -9 `ps auxww | grep \'celery worker\' | awk \'{print $2}\'`')


@task
def start_debug_server(host="0.0.0.0", port=8000):
    stop_celery()
    start_celery()
    local('python manage.py runserver {}:{}'.format(host, port))


@task
def stop_all():
    stop_celery()
    stop_rabbitmq()


def parse_bool(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        return value.lower() == 'true'
    else:
        raise Exception('Cannot convert {} to bool'.format(type(value)))


def log_status(status, task_name):
    LOG.info("%s status: %s", task_name, STATUS._fields[STATUS.index(status)])


@task
def reset_website():
    # WARNING: destroys the existing website and creates with all
    # of the required inital data loaded (e.g., the KnobCatalog)

    # Recreate the ottertune database
    user = DATABASES['default']['USER']
    passwd = DATABASES['default']['PASSWORD']
    name = DATABASES['default']['NAME']
    local("mysql -u {} -p{} -N -B -e \"DROP DATABASE IF EXISTS {}\"".format(
        user, passwd, name))
    local("mysql -u {} -p{} -N -B -e \"CREATE DATABASE {}\"".format(
        user, passwd, name))

    # Reinitialize the website
    local('python manage.py migrate website')
    local('python manage.py migrate')


@task
def create_test_website():
    # WARNING: destroys the existing website and creates a new one. Creates
    # a test user and two test sessions: a basic session and a tuning session.
    # The tuning session has knob/metric data preloaded (5 workloads, 20
    # samples each).
    reset_website()
    local("python manage.py loaddata test_website.json")


@task
def setup_test_user():
    # Adds a test user to an existing website with two empty sessions
    local(("echo \"from django.contrib.auth.models import User; "
           "User.objects.filter(email='user@email.com').delete(); "
           "User.objects.create_superuser('user', 'user@email.com', 'abcd123')\" "
           "| python manage.py shell"))

    local("python manage.py loaddata test_user_sessions.json")


@task
def generate_and_load_data(n_workload, n_samples_per_workload, upload_code,
                           random_seed=''):
    local('python script/controller_simulator/data_generator.py {} {} {}'.format(
        n_workload, n_samples_per_workload, random_seed))
    local(('python script/controller_simulator/upload_data.py '
           'script/controller_simulator/generated_data {}').format(upload_code))


@task
def dumpdata(dumppath):
    # Helper function for calling Django's loaddata function that excludes
    # the static fixture data from being dumped
    excluded_models = ['DBMSCatalog', 'KnobCatalog', 'MetricCatalog', 'Hardware']
    cmd = 'python manage.py dumpdata --natural-foreign --natural-primary'
    for model in excluded_models:
        cmd += ' --exclude website.' + model
    cmd += ' > ' + dumppath
    local(cmd)


@task
def run_background_tasks():
    # Runs the background tasks just once.
    cmd = ("from website.tasks import run_background_tasks; "
           "run_background_tasks()")
    local(('export PYTHONPATH={}\:$PYTHONPATH; '  # pylint: disable=anomalous-backslash-in-string
           'django-admin shell --settings=website.settings '
           '-c\"{}\"').format(PROJECT_ROOT, cmd))

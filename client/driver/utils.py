import importlib
import os

from fabric.api import hide, local, settings, task
from fabric.api import get as _get, put as _put, run as _run, sudo as _sudo

dconf = None


def load_driver_conf():
    driver_conf = os.environ.get('DRIVER_CONFIG', 'driver_config')
    if driver_conf.endswith('.py'):
        driver_conf = driver_conf[:-len('.py')]
    dmod = importlib.import_module(driver_conf)
    global dconf
    if not dconf:
        dconf = dmod
    return dmod


def parse_bool(value):
    if not isinstance(value, bool):
        value = str(value).lower() == 'true'
    return value


@task
def run(cmd, **kwargs):
    try:
        if dconf.HOST_CONN == 'remote':
            res = _run(cmd, **kwargs)
        elif dconf.HOST_CONN == 'local':
            res = local(cmd, capture=True, **kwargs)
        else:  # docker
            opts = ''
            cmdd = cmd
            if cmd.endswith('&'):
                cmdd = cmd[:-1].strip()
                opts = '-d '
            res = local('docker exec {} -ti {} /bin/bash -c "{}"'.format(
                opts, dconf.CONTAINER_NAME, cmdd),
                capture=True, **kwargs)
    except TypeError as e:
        err = str(e).strip()
        if 'unexpected keyword argument' in err:
            offender = err.rsplit(' ', 1)[-1][1:-1]
            kwargs.pop(offender)
            res = run(cmd, **kwargs)
        else:
            raise e
    return res


@task
def sudo(cmd, user=None, **kwargs):
    if dconf.HOST_CONN == 'remote':
        res = _sudo(cmd, user=user, **kwargs)

    elif dconf.HOST_CONN == 'local':
        pre_cmd = 'sudo '
        if user:
            pre_cmd += '-u {} '.format(user)
        res = local(pre_cmd + cmd, capture=True, **kwargs)

    else:  # docker
        user = user or 'root'
        opts = '-ti -u {}'.format(user or 'root')
        if user == 'root':
            opts += ' -w /'
        res = local('docker exec {} {} /bin/bash -c "{}"'.format(
            opts, dconf.CONTAINER_NAME, cmd), capture=True)

    return res


@task
def get(remote_path, local_path, use_sudo=False):
    use_sudo = parse_bool(use_sudo)

    if dconf.HOST_CONN == 'remote':
        res = _get(remote_path, local_path, use_sudo=use_sudo)
    elif dconf.HOST_CONN == 'local':
        pre_cmd = 'sudo ' if use_sudo else ''
        opts = '-r' if os.path.isdir(remote_path) else ''
        res = local('{}cp {} {} {}'.format(pre_cmd, opts, remote_path, local_path))
    else:  # docker
        res = local('docker cp {}:{} {}'.format(dconf.CONTAINER_NAME, remote_path, local_path))
    return res


@task
def put(local_path, remote_path, use_sudo=False):
    use_sudo = parse_bool(use_sudo)

    if dconf.HOST_CONN == 'remote':
        res = _put(local_path, remote_path, use_sudo=use_sudo)
    elif dconf.HOST_CONN == 'local':
        pre_cmd = 'sudo ' if use_sudo else ''
        opts = '-r' if os.path.isdir(local_path) else ''
        res = local('{}cp {} {} {}'.format(pre_cmd, opts, local_path, remote_path))
    else:  # docker
        res = local('docker cp {} {}:{}'.format(local_path, dconf.CONTAINER_NAME, remote_path))
    return res


@task
def file_exists(filename):
    with settings(warn_only=True), hide('warnings'):
        res = run('[ -f {} ]'.format(filename))
    return res.return_code == 0

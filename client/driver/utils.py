import importlib
import os

from fabric.api import hide, local, settings, task
from fabric.api import get as _get, put as _put, run as _run, sudo as _sudo

dconf = None  # pylint: disable=invalid-name


def load_driver_conf():
    # The default config file is 'driver_config.py' but you can use
    # set the env 'DRIVER_CONFIG' to the path of a different config
    # file to override it.
    global dconf  # pylint: disable=global-statement,invalid-name
    if not dconf:
        driver_conf = os.environ.get('DRIVER_CONFIG', 'driver_config')
        if driver_conf.endswith('.py'):
            driver_conf = driver_conf[:-len('.py')]
        mod = importlib.import_module(driver_conf)
        dconf = mod

        # Generate the login string of the host connection
        if dconf.HOST_CONN == 'local':
            login_str = 'localhost'

        elif dconf.HOST_CONN in ['remote', 'remote_docker']:
            if not dconf.LOGIN_HOST:
                raise ValueError("LOGIN_HOST must be set if HOST_CONN=remote")

            login_str = dconf.LOGIN_HOST
            if dconf.LOGIN_NAME:
                login_str = '{}@{}'.format(dconf.LOGIN_NAME, login_str)

            if dconf.LOGIN_PORT:
                login_str += ':{}'.format(dconf.LOGIN_PORT)

        elif dconf.HOST_CONN == 'docker':
            if not dconf.CONTAINER_NAME:
                raise ValueError("CONTAINER_NAME must be set if HOST_CONN=docker")
            login_str = 'localhost'

        else:
            raise ValueError(("Invalid HOST_CONN: {}. Valid values are "
                              "'local', 'remote', or 'docker'.").format(dconf.HOST_CONN))
        dconf.LOGIN = login_str

    return dconf


def parse_bool(value):
    if not isinstance(value, bool):
        value = str(value).lower() == 'true'
    return value


def get_content(response):
    content_type = response.headers.get('Content-Type', '')
    if content_type == 'application/json':
        content = response.json()
    else:
        content = response.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
    return content


@task
def run(cmd, capture=True, remote_only=False, **kwargs):
    capture = parse_bool(capture)

    try:
        if dconf.HOST_CONN == 'remote':
            res = _run(cmd, **kwargs)
        elif dconf.HOST_CONN == 'local':
            res = local(cmd, capture=capture, **kwargs)
        else:  # docker or remote_docker
            opts = ''
            cmdd = cmd
            if cmd.endswith('&'):
                cmdd = cmd[:-1].strip()
                opts = '-d '
            if remote_only:
                docker_cmd = cmdd
            else:
                docker_cmd = 'docker exec {} -ti {} /bin/bash -c "{}"'.format(
                    opts, dconf.CONTAINER_NAME, cmdd)
            if dconf.HOST_CONN == 'docker':
                res = local(docker_cmd, capture=capture, **kwargs)
            elif dconf.HOST_CONN == 'remote_docker':
                res = _run(docker_cmd, **kwargs)
            else:
                raise Exception('wrong HOST_CONN type {}'.format(dconf.HOST_CONN))
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
def sudo(cmd, user=None, capture=True, remote_only=False, **kwargs):
    capture = parse_bool(capture)

    if dconf.HOST_CONN == 'remote':
        res = _sudo(cmd, user=user, **kwargs)

    elif dconf.HOST_CONN == 'local':
        pre_cmd = 'sudo '
        if user:
            pre_cmd += '-u {} '.format(user)
        res = local(pre_cmd + cmd, capture=capture, **kwargs)

    else:  # docker or remote_docker
        user = user or 'root'
        opts = '-ti -u {}'.format(user or 'root')
        if user == 'root':
            opts += ' -w /'
        if remote_only:
            docker_cmd = cmd
        else:
            docker_cmd = 'docker exec {} {} /bin/bash -c "{}"'.format(
                opts, dconf.CONTAINER_NAME, cmd)
        if dconf.HOST_CONN == 'docker':
            res = local(docker_cmd, capture=capture, **kwargs)
        elif dconf.HOST_CONN == 'remote_docker':
            res = _sudo(docker_cmd, **kwargs)
        else:
            raise Exception('wrong HOST_CONN type {}'.format(dconf.HOST_CONN))
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
    else:  # docker or remote_docker
        docker_cmd = 'docker cp -L {}:{} {}'.format(dconf.CONTAINER_NAME, remote_path, local_path)
        if dconf.HOST_CONN == 'docker':
            if dconf.DB_CONF_MOUNT is True:
                pre_cmd = 'sudo ' if use_sudo else ''
                opts = '-r' if os.path.isdir(remote_path) else ''
                res = local('{}cp {} {} {}'.format(pre_cmd, opts, remote_path, local_path))
            else:
                res = local(docker_cmd)
        elif dconf.HOST_CONN == 'remote_docker':
            if dconf.DB_CONF_MOUNT is True:
                res = _get(remote_path, local_path, use_sudo=use_sudo)
            else:
                res = sudo(docker_cmd, remote_only=True)
                res = _get(local_path, local_path, use_sudo)
        else:
            raise Exception('wrong HOST_CONN type {}'.format(dconf.HOST_CONN))
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
    else:  # docker or remote_docker
        docker_cmd = 'docker cp -L {} {}:{}'.format(local_path, dconf.CONTAINER_NAME, remote_path)
        if dconf.HOST_CONN == 'docker':
            if dconf.DB_CONF_MOUNT is True:
                pre_cmd = 'sudo ' if use_sudo else ''
                opts = '-r' if os.path.isdir(local_path) else ''
                res = local('{}cp {} {} {}'.format(pre_cmd, opts, local_path, remote_path))
            else:
                res = local(docker_cmd)
        elif dconf.HOST_CONN == 'remote_docker':
            if dconf.DB_CONF_MOUNT is True:
                res = _put(local_path, remote_path, use_sudo=use_sudo)
            else:
                res = _put(local_path, local_path, use_sudo=True)
                res = sudo(docker_cmd, remote_only=True)
        else:
            raise Exception('wrong HOST_CONN type {}'.format(dconf.HOST_CONN))
    return res


@task
def run_sql_script(scriptfile, *args):
    if dconf.DB_TYPE == 'oracle':
        if dconf.HOST_CONN != 'local':
            scriptdir = '/home/oracle/oracleScripts'
            remote_path = os.path.join(scriptdir, scriptfile)
            if not file_exists(remote_path):
                run('mkdir -p {}'.format(scriptdir))
                put(os.path.join('./oracleScripts', scriptfile), remote_path)
                sudo('chown -R oracle:oinstall /home/oracle/oracleScripts')
        res = run('sh {} {}'.format(remote_path, ' '.join(args)))
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))
    return res


@task
def file_exists(filename):
    with settings(warn_only=True), hide('warnings'):  # pylint: disable=not-context-manager
        res = run('[ -f {} ]'.format(filename))
    return res.return_code == 0


@task
def dir_exists(dirname):
    with settings(warn_only=True), hide('warnings'):  # pylint: disable=not-context-manager
        res = run('[ -d {} ]'.format(dirname))
    return res.return_code == 0


class FabricException(Exception):
    pass

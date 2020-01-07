#
# OtterTune - runcelery.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import os

from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.utils import autoreload
from fabric.api import hide, local, settings


class Command(BaseCommand):
    help = 'Start celery and celerybeat using the auto-reloader.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--loglevel',
            metavar='LOGLEVEL',
            help='Logging level, choose between DEBUG, INFO, WARNING, ERROR, CRITICAL, or FATAL. '
                 'Defaults to DEBUG if settings.DEBUG is true, otherwise INFO.')
        parser.add_argument(
            '--pool',
            metavar='POOL_CLS',
            default='threads',
            help='Pool implementation: prefork (default), eventlet, gevent, solo or threads. '
                 'Default: threads.')
        parser.add_argument(
            '--celery-pidfile',
            metavar='PIDFILE',
            default='celery.pid',
            help='File used to store the process pid. The program will not start if this '
                 'file already exists and the pid is still alive. Default: celery.pid.')
        parser.add_argument(
            '--celerybeat-pidfile',
            metavar='PIDFILE',
            default='celerybeat.pid',
            help='File used to store the process pid. The program will not start if this '
                 'file already exists and the pid is still alive. Default: celerybeat.pid.')
        parser.add_argument(
            '--celery-options',
            metavar='OPTIONS',
            help="A comma-separated list of additional options to pass to celery, "
                 "see 'python manage.py celery worker --help' for all available options. "
                 "IMPORTANT: the option's initial -/-- must be omitted. "
                 "Example: '--celery-options purge,include=foo.tasks,q'.")
        parser.add_argument(
            '--celerybeat-options',
            metavar='OPTIONS',
            help="A comma-separated list of additional options to pass to celerybeat, "
                 "see 'python manage.py celerybeat --help' for all available options. "
                 "IMPORTANT: the option's initial -/-- must be omitted. "
                 "Example: '--celerybeat-options uid=123,q'.")

    def inner_run(self, *args, **options):  # pylint: disable=unused-argument
        autoreload.raise_last_exception()

        for pidfile in (options['celery_pidfile'], options['celerybeat_pidfile']):
            if os.path.exists(pidfile):
                with open(pidfile, 'r') as f:
                    pid = f.read().strip()
                with settings(warn_only=True), hide('commands'):  # pylint: disable=not-context-manager
                    local('kill -9 {}'.format(pid))
                    local('rm -f {}'.format(pidfile))
        call_command('startcelery', silent=True, pipe='', **options)
        self.stdout.write(self.style.SUCCESS("Successfully reloaded celery and celerybeat."))

    def handle(self, *args, **options):
        autoreload.main(self.inner_run, None, options)

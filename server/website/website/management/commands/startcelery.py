#
# OtterTune - startcelery.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import os
import time

from django.conf import settings
from django.core.management.base import BaseCommand
from fabric.api import hide, lcd, local


class Command(BaseCommand):
    help = 'Start celery and celerybeat in the background.'
    celery_cmd = 'python3 manage.py {cmd} {opts} {pipe} &'.format
    max_wait_sec = 15

    def add_arguments(self, parser):
        parser.add_argument(
            '--celery-only',
            action='store_true',
            help='Start celery only (skip celerybeat).')
        parser.add_argument(
            '--celerybeat-only',
            action='store_true',
            help='Start celerybeat only (skip celery).')
        parser.add_argument(
            '--loglevel',
            metavar='LOGLEVEL',
            help='Logging level, choose between DEBUG, INFO, WARNING, ERROR, CRITICAL, or FATAL. '
                 'Defaults to DEBUG if settings.DEBUG is true, otherwise INFO.')
        parser.add_argument(
            '--pool',
            metavar='POOL_CLS',
            default='threads',
            help='Pool implementation: prefork (default), eventlet, gevent, solo or threads.'
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

    def _parse_suboptions(self, suboptions):
        suboptions = suboptions or ''
        parsed = []
        for opt in suboptions.split(','):
            if opt:
                opt = ('-{}' if len(opt) == 1 else '--{}').format(opt)
                parsed.append(opt)
        return parsed

    def handle(self, *args, **options):
        loglevel = options['loglevel'] or ('DEBUG' if settings.DEBUG else 'INFO')
        celery_options = [
            '--loglevel={}'.format(loglevel),
            '--pool={}'.format(options['pool']),
            '--pidfile={}'.format(options['celery_pidfile']),
        ] + self._parse_suboptions(options['celery_options'])
        celerybeat_options = [
            '--loglevel={}'.format(loglevel),
            '--pidfile={}'.format(options['celerybeat_pidfile']),
        ] + self._parse_suboptions(options['celerybeat_options'])

        pipe = '' if 'console' in settings.LOGGING['loggers']['celery']['handlers'] \
            else '> /dev/null 2>&1'

        with lcd(settings.PROJECT_ROOT), hide('commands'):
            if not options['celerybeat_only']:
                local(self.celery_cmd(
                    cmd='celery worker', opts=' '.join(celery_options), pipe=pipe))

            if not options['celery_only']:
                local(self.celery_cmd(
                    cmd='celerybeat', opts=' '.join(celerybeat_options), pipe=pipe))

        pidfiles = [options['celery_pidfile'], options['celerybeat_pidfile']]
        wait_sec = 0

        while wait_sec < self.max_wait_sec and len(pidfiles) > 0:
            time.sleep(1)
            wait_sec += 1

            for i in range(len(pidfiles))[::-1]:
                if os.path.exists(pidfiles[i]):
                    pidfiles.pop(i)

        for name in ('celery', 'celerybeat'):
            if os.path.exists(options[name + '_pidfile']):
                self.stdout.write(self.style.SUCCESS(
                    "Successfully started '{}'.".format(name)))
            else:
                self.stdout.write(self.style.NOTICE(
                    "Failed to start '{}'.".format(name)))

#
# OtterTune - stopcelery.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import os
import time

from django.core.management.base import BaseCommand
from fabric.api import local


class Command(BaseCommand):
    help = 'Stop celery and celerybeat and remove pid files.'
    celery_cmd = 'python3 manage.py {cmd} {opts} &'.format
    max_wait_sec = 15

    def add_arguments(self, parser):
        parser.add_argument(
            '--celery-pidfile',
            metavar='PIDFILE',
            default='celery.pid',
            help="Alternate path to the process' pid file if not located at ./celery.pid.")
        parser.add_argument(
            '--celerybeat-pidfile',
            metavar='PIDFILE',
            default='celerybeat.pid',
            help="Alternate path to the process' pid file if not located at ./celerybeat.pid.")

    def handle(self, *args, **options):
        check_pidfiles = []
        for name in ('celery', 'celerybeat'):
            try:
                pidfile = options[name + '_pidfile']
                with open(pidfile, 'r') as f:
                    pid = f.read()
                local('kill {}'.format(pid))
                check_pidfiles.append((name, pidfile))
            except Exception as e:
                self.stdout.write(self.style.NOTICE(
                    "ERROR: an exception occurred while stopping '{}':\n{}\n".format(name, e)))

        if check_pidfiles:
            self.stdout.write("Waiting for processes to shutdown...\n")
        for name, pidfile in check_pidfiles:
            wait_sec = 0
            while os.path.exists(pidfile) and wait_sec < self.max_wait_sec:
                time.sleep(1)
                wait_sec += 1
            if os.path.exists(pidfile):
                self.stdout.write(self.style.NOTICE(
                    "WARNING: file '{}' still exists after stopping {}.".format(
                        pidfile, name)))
            else:
                self.stdout.write(self.style.SUCCESS(
                    "Successfully stopped '{}'.".format(name)))

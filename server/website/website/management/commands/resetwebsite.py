#
# OtterTune - resetwebsite.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.core.management.base import BaseCommand
from fabric.api import local
from website.settings import DATABASES


class Command(BaseCommand):
    help = 'dump the website; reset the website; load data from file if specified.'

    def add_arguments(self, parser):
        parser.add_argument(
            '-d', '--dumpfile',
            metavar='FILE',
            help='Name of the file to dump data to. '
                 'Default: dump_website.json')
        parser.add_argument(
            '-l', '--loadfile',
            metavar='FILE',
            help='Name of the file to load data from. ')

    def reset_website(self):
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
        local('python manage.py migrate')

    def handle(self, *args, **options):
        dumpfile = options['dumpfile'] if options['dumpfile'] else 'dump_website.json'
        local("python manage.py dumpdata admin auth django_db_logger djcelery sessions\
               sites website > {}".format(dumpfile))
        self.reset_website()
        if options['loadfile']:
            local("python manage.py loaddata '{}'".format(options['loadfile']))

        self.stdout.write(self.style.SUCCESS(
            "Successfully reset website."))

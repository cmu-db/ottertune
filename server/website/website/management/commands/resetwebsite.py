#
# OtterTune - resetwebsite.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django import db
from django.core.management import call_command
from django.core.management.base import BaseCommand
from fabric.api import local
from website.settings import DATABASES


class Command(BaseCommand):
    help = 'dump the website; reset the website; load data from file if specified.'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        engine = DATABASES['default']['ENGINE']
        user = DATABASES['default']['USER']
        passwd = DATABASES['default']['PASSWORD']
        host = DATABASES['default']['HOST']
        port = DATABASES['default']['PORT']

        if engine.endswith('mysql'):
            db_cmd_fmt = 'mysql -u {user} -p -h {host} -P {port} -N -B -e "{{cmd}}"'
        elif engine.endswith('postgresql'):
            db_cmd_fmt = 'PGPASSWORD={passwd} psql -U {user} -h {host} -p {port} -c "{{cmd}}"'
        else:
            raise NotImplementedError("Database engine '{}' is not implemented.".format(engine))

        self._db_cmd_fmt = db_cmd_fmt.format(user=user, passwd=passwd, host=host, port=port).format

    def call_db_command(self, cmd):
        local(self._db_cmd_fmt(cmd=cmd))

    def add_arguments(self, parser):
        parser.add_argument(
            '-d', '--dumpfile',
            metavar='FILE',
            default='dump_website.json',
            help="Name of the file to dump data to. "
                 "Default: 'dump_website.json'")
        parser.add_argument(
            '-l', '--loadfile',
            metavar='FILE',
            help="Name of the file to load data from. "
                 "Default: '' (no data loaded)")

    def reset_website(self):
        # WARNING: destroys the existing website and creates with all
        # of the required inital data loaded (e.g., the KnobCatalog)

        # Recreate the ottertune database
        db.connections.close_all()
        dbname = DATABASES['default']['NAME']
        self.call_db_command("DROP DATABASE IF EXISTS {}".format(dbname))
        self.call_db_command("CREATE DATABASE {}".format(dbname))

        # Reinitialize the website
        call_command('makemigrations', 'website')
        call_command('migrate')
        call_command('startcelery')

    def handle(self, *args, **options):
        call_command('dumpwebsite', dumpfile=options['dumpfile'])
        call_command('stopcelery')

        self.reset_website()

        loadfile = options['loadfile']
        if loadfile:
            self.stdout.write("Loading database from file '{}'...".format(loadfile))
            call_command('loaddata', loadfile)

        self.stdout.write(self.style.SUCCESS("Successfully reset website."))

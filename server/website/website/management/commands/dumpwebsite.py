#
# OtterTune - dumpwebsite.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.core.management import call_command
from django.core.management.base import BaseCommand
from fabric.api import hide, local


class Command(BaseCommand):
    help = 'dump the website.'

    def add_arguments(self, parser):
        parser.add_argument(
            '-d', '--dumpfile',
            metavar='FILE',
            default='dump_website.json',
            help="Name of the file to dump data to. "
                 "Default: 'dump_website.json[.gz]'")
        parser.add_argument(
            '--compress',
            action='store_true',
            help='Compress dump data (gzip). Default: False')

    def handle(self, *args, **options):
        dumpfile = options['dumpfile']
        compress = options['compress']
        if compress:
            if dumpfile.endswith('.gz'):
                dstfile = dumpfile
                dumpfile = dumpfile[:-len('.gz')]
            else:
                dstfile = dumpfile + '.gz'
        else:
            dstfile = dumpfile

        self.stdout.write("Dumping database to file '{}'...".format(dstfile))
        call_command('dumpdata', 'admin', 'auth', 'django_db_logger', 'djcelery', 'sessions',
                     'sites', 'website', output=dumpfile)

        if compress:
            with hide("commands"):  # pylint: disable=not-context-manager
                local("gzip {}".format(dumpfile))

        self.stdout.write(self.style.SUCCESS(
            "Successfully dumped website to '{}'.".format(dstfile)))

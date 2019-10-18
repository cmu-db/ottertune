#
# OtterTune - dumpdebuginfo.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import os

from django.core.management.base import BaseCommand, CommandError

from website.models import Session
from website.utils import dump_debug_info


class Command(BaseCommand):
    help = 'Dump debug information for the session with the given upload code.'

    def add_arguments(self, parser):
        parser.add_argument(
            'uploadcode',
            metavar='UPLOADCODE',
            help="The session's upload code to.")
        parser.add_argument(
            '-f', '--filename',
            metavar='FILE',
            help='Name of the file to write the debug information. '
                 'Default: debug_[timestamp].tar.gz')
        parser.add_argument(
            '-d', '--directory',
            metavar='DIR',
            help='Path of the directory to write the debug information to. '
                 'Default: current directory')
        parser.add_argument(
            '--prettyprint',
            action='store_true',
            help='Pretty print the output.')

    def handle(self, *args, **options):
        directory = options['directory'] or ''
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        try:
            session = Session.objects.get(upload_code=options['uploadcode'])
        except Session.DoesNotExist:
            raise CommandError(
                "ERROR: Session with upload code '{}' not exist.".format(options['uploadcode']))

        debug_info, root = dump_debug_info(session, pretty_print=options['prettyprint'])

        filename = options['filename'] or root
        if not filename.endswith('.tar.gz'):
            filename += '.tar.gz'
        path = os.path.join(directory, filename)

        with open(path, 'wb') as f:
            f.write(debug_info.getvalue())

        self.stdout.write(self.style.SUCCESS(
            "Successfully dumped debug information to '{}'.".format(path)))

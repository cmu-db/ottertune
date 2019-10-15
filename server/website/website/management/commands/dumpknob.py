#
# OtterTune - dumpknob.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import json
import os

from django.core.management.base import BaseCommand, CommandError

from website.models import Session, SessionKnob, SessionKnobManager


class Command(BaseCommand):
    help = 'Dump knobs for the session with the given upload code.'

    def add_arguments(self, parser):
        parser.add_argument(
            'uploadcode',
            metavar='UPLOADCODE',
            help="The session's upload code.")
        parser.add_argument(
            '-f', '--filename',
            metavar='FILE',
            help='Name of the file to write the session knob tunability to. '
                 'Default: knob.json')
        parser.add_argument(
            '-d', '--directory',
            metavar='DIR',
            help='Path of the directory to write the session knob tunability to. '
                 'Default: current directory')

    def handle(self, *args, **options):
        directory = options['directory'] or ''
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        try:
            session = Session.objects.get(upload_code=options['uploadcode'])
        except Session.DoesNotExist:
            raise CommandError(
                "ERROR: Session with upload code '{}' not exist.".format(options['uploadcode']))

        session_knobs = SessionKnobManager.get_knob_min_max_tunability(session)

        filename = options['filename'] or 'knobs.json'
        path = os.path.join(directory, filename)

        with open(path, 'w') as f:
            json.dump(session_knobs, f, indent=4)

        self.stdout.write(self.style.SUCCESS(
            "Successfully dumped knob information to '{}'.".format(path)))

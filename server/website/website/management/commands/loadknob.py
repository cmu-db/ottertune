#
# OtterTune - loadknob.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import json
import os

from django.core.management.base import BaseCommand, CommandError

from website.models import Session, SessionKnob, SessionKnobManager


class Command(BaseCommand):
    help = 'load knobs for the session with the given upload code.'

    def add_arguments(self, parser):
        parser.add_argument(
            'uploadcode',
            metavar='UPLOADCODE',
            help="The session's upload code.")
        parser.add_argument(
            '-f', '--filename',
            metavar='FILE',
            help='Name of the file to read the session knob tunability from. '
                 'Default: knob.json')
        parser.add_argument(
            '-d', '--directory',
            metavar='DIR',
            help='Path of the directory to read the session knob tunability from. '
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

        filename = options['filename'] or 'knobs.json'
        path = os.path.join(directory, filename)

        with open(path, 'r') as f:
            knobs = json.load(f)

        SessionKnobManager.set_knob_min_max_tunability(session, knobs)

        self.stdout.write(self.style.SUCCESS(
            "Successfully load knob information from '{}'.".format(path)))

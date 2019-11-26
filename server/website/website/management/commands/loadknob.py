#
# OtterTune - loadknob.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import json
import os
from argparse import RawTextHelpFormatter

from django.core.management.base import BaseCommand, CommandError

from website.models import Session, SessionKnob, SessionKnobManager

HELP = """
Load knobs for the session with the given upload code.

example of JSON file format:
  {
      "global.knob1": {
          "minval": 0,
          "maxval": 100,
          "tunable": true
      },
      "global.knob2": {
          "minval": 1000000,
          "maxval": 2000000,
          "tunable": false
      }
  }
"""


class Command(BaseCommand):
    help = HELP

    def create_parser(self, prog_name, subcommand):
        parser = super(Command, self).create_parser(prog_name, subcommand)
        parser.formatter_class = RawTextHelpFormatter
        return parser

    def add_arguments(self, parser):
        parser.add_argument(
            'uploadcode',
            metavar='UPLOADCODE',
            help="The session's upload code.")
        parser.add_argument(
            '-f', '--filename',
            metavar='FILE',
            default='session_knobs.json',
            help='Name of the file to read the session knob tunability from. '
                 'Default: session_knobs.json')
        parser.add_argument(
            '-d', '--directory',
            metavar='DIR',
            help='Path of the directory to read the session knob tunability from. '
                 'Default: current directory')
        parser.add_argument(
            '--disable-others',
            action='store_true',
            help='Disable the knob tunability of all session knobs NOT included '
                 'in the JSON file. Default: False')

    def handle(self, *args, **options):
        directory = options['directory'] or ''
        path = os.path.join(directory, options['filename'])

        try:
            with open(path, 'r') as f:
                knobs = json.load(f)
        except FileNotFoundError:
            raise CommandError("ERROR: File '{}' does not exist.".format(path))
        except json.decoder.JSONDecodeError:
            raise CommandError("ERROR: Unable to decode JSON file '{}'.".format(path))

        try:
            session = Session.objects.get(upload_code=options['uploadcode'])
        except Session.DoesNotExist:
            raise CommandError(
                "ERROR: Session with upload code '{}' not exist.".format(options['uploadcode']))

        SessionKnobManager.set_knob_min_max_tunability(
            session, knobs, disable_others=options['disable_others'])

        self.stdout.write(self.style.SUCCESS(
            "Successfully load knob information from '{}'.".format(path)))

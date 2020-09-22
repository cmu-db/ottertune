#
# OtterTune - updateknob.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import json
import os
from argparse import RawTextHelpFormatter

from django.core.management.base import BaseCommand, CommandError


HELP = """
Update the knobs in a json file.

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


def update_knobs(tunable_knob_file, knobs, num):
    if num <= 0:
        return
    cnt = 0
    with open(tunable_knob_file, 'r') as f:
        # csv file columns: knob,min,max,vartype,safe,note,
        # discard the label row
        lines = f.readlines()[1:-1]
        for line in lines:
            field_list = line.split(',')
            name = ('global.' + field_list[0]).lower()
            if name in knobs:
                knob = knobs[name]
                if knob['tunable'] is True:
                    continue
                knob['tunable'] = True
                if len(field_list[1]) > 0 and len(field_list[2]) > 0:
                    knob['minval'] = field_list[1]
                    knob['maxval'] = field_list[2]
                cnt += 1
                if cnt == num:
                    break


class Command(BaseCommand):
    help = HELP

    def create_parser(self, prog_name, subcommand):
        parser = super(Command, self).create_parser(prog_name, subcommand)
        parser.formatter_class = RawTextHelpFormatter
        return parser

    def add_arguments(self, parser):
        parser.add_argument(
            '-f', '--filename',
            metavar='FILE',
            default='session_knobs.json',
            help='Name of the target knob file in json format. '
                 'Default: session_knobs.json')
        parser.add_argument(
            '-s', '--source',
            metavar='SOURCE',
            default='selected_knobs.csv',
            help='Name of the file to read the session knob tunability from. '
                 'Default: selected_knobs.csv')
        parser.add_argument(
            '-o', '--output',
            metavar='OUTPUT',
            default='selected_knobs.csv',
            help='Name of the file to write the updated session knob tunability to. '
                 'Default: selected_knobs.csv')
        parser.add_argument(
            '-d', '--file_dir',
            metavar='FILEDIR',
            help='Path of the directory of the target knob file. '
                 'Default: current directory')
        parser.add_argument(
            '-i', '--source_dir',
            metavar='SOURCEDIR',
            help='Path of the directory to read the session knob tunability from. '
                 'Default: current directory')
        parser.add_argument(
            '-n', '--num',
            metavar='NUM',
            default='50',
            help='Total number of tunable session knobs. '
                 'Default: 50')

    def handle(self, *args, **options):
        file_dir = options['file_dir'] or ''
        path = os.path.join(file_dir, options['filename'])

        try:
            with open(path, 'r') as f:
                knobs = json.load(f)
        except FileNotFoundError:
            raise CommandError("ERROR: File '{}' does not exist.".format(path))
        except json.decoder.JSONDecodeError:
            raise CommandError("ERROR: Unable to decode JSON file '{}'.".format(path))

        source_dir = options['source_dir'] or ''
        tunable_knob_file = os.path.join(source_dir, options['source'])
        target_tunable_knobs = int(options['num'])

        cur_tunable_knobs = 0
        for knob in knobs.values():
            if knob['tunable']:
                cur_tunable_knobs += 1
        if cur_tunable_knobs < target_tunable_knobs:
            update_knobs(tunable_knob_file, knobs, target_tunable_knobs - cur_tunable_knobs)
            cur_tunable_knobs = 0
            for knob in knobs.values():
                if knob['tunable']:
                    cur_tunable_knobs += 1

        out_path = os.path.join(file_dir, options['output'])
        with open(out_path, 'w') as f:
            json.dump(knobs, f, indent=4)
        self.stdout.write(self.style.SUCCESS(
            "Writing {} tunable session knobs into {}.".format(cur_tunable_knobs, out_path)))

#!/usr/bin/env python
# encoding: utf-8
#
# OtterTune - source_validator.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

# ==============================================
# SOURCE VALIDATOR
# ==============================================
#
# Adapted from the source validator used by Peloton.
# (see https://github.com/cmu-db/peloton/blob/master/script/validators/source_validator.py)

import argparse
import logging
import imp
import os
import re
import subprocess
import sys
import json
import functools
from collections import namedtuple
from fabric.api import local, settings, quiet

EXIT_SUCCESS = 0
EXIT_FAILURE = -1

# ==============================================
# CONFIGURATION
# ==============================================

# Logging
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler())
LOG.setLevel(logging.INFO)

# NOTE: the absolute path to ottertune directory is calculated from current
# directory structure: ottertune/server/website/scripts/validators/<this_file>
# OTTERTUNE_DIR needs to be redefined if the directory structure is changed.
CODE_SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
OTTERTUNE_DIR = os.path.abspath(functools.reduce(os.path.join,
                                                 [CODE_SOURCE_DIR,
                                                  os.path.pardir,
                                                  os.path.pardir]))

# Other directory paths used are relative to OTTERTUNE_DIR
DEFAULT_DIRS = [
    os.path.join(OTTERTUNE_DIR, "server")
]

# Directories that should NOT be checked
EXCLUDE_DIRECTORIES = [
    # Django-generated directories
    os.path.join(OTTERTUNE_DIR, "server/website/website/migrations"),
]

# Files that should NOT be checked
EXCLUDE_FILES = [
    # Django-generated files
    os.path.join(OTTERTUNE_DIR, 'server/website/manage.py'),

    # Untracked files
    os.path.join(OTTERTUNE_DIR, 'server/website/website/settings/credentials.py'),

    # TODO (dvanaken): remove this and format files after merging parser tests
    os.path.join(OTTERTUNE_DIR, "server/website/website/utils.py"),

    # TODO (dvanaken): remove these and format files after merging #62 (parser checks) and
    # parser tests (for postgres.py only)
    os.path.join(OTTERTUNE_DIR, "server/website/website/parser/base.py"),
    os.path.join(OTTERTUNE_DIR, "server/website/website/parser/postgres.py"),
]

# Regex patterns
PYCODESTYLE_COMMENT_PATTERN = re.compile(r'#\s*pycodestyle:\s*disable\s*=\s*[\w\,\s]+$')
ILLEGAL_PATTERNS = [
    (re.compile(r'^print[ (]'), "Do not use 'print'. Use the logging module instead.")
]
HEADER_PATTERN = re.compile(r'#\n#.*\n#\n# Copyright.*\n#\n')

# Stdout format strings
SEPARATOR = 80 * '-'
OUTPUT_FMT = (
    '' + SEPARATOR + '\n\n'
    '\033[1m'        # start bold text
    '%s\n'
    'FAILED: %s\n\n'
    '\033[0m'        # end bold text
    '%s'
)
VALIDATOR_FMT = '{name}\n{u}\n{out}'.format
MSG_PREFIX_FMT = ' {filename}:{line:3d}: '.format
MSG_SUFFIX_FMT = ' ({symbol})'.format


# ==============================================
# UTILITY FUNCTION DEFINITIONS
# ==============================================

def format_message(filename, line, message, symbol=None):
    out_prefix = MSG_PREFIX_FMT(filename=filename, line=line)
    out_suffix = '' if symbol is None else MSG_SUFFIX_FMT(symbol=symbol)

    # Crop the message details to make the output more readable
    max_msg_len = 80 - len(out_prefix) - len(out_suffix)
    if len(message) > max_msg_len:
        message = message[:max_msg_len - 3] + '...'
    output = (out_prefix + message + out_suffix).replace('\n', '')
    return output + '\n'


def validate_validator(modules, config_path):
    status = True

    # Check if required modules are installed
    for module in modules:
        if module is not None:
            try:
                imp.find_module(module)
            except ImportError:
                LOG.error("Cannot find module %s", module)
                status = False

    # Check that the config file exists if assigned
    if config_path is not None and not os.path.isfile(config_path):
        LOG.error("Cannot find config file %s", config_path)
        status = False
    return status


# Validate the file passed as argument
def validate_file(file_path):
    if file_path in EXCLUDE_FILES or not file_path.endswith(".py"):
        status = True
    else:
        LOG.debug("Validating file: %s", file_path)
        status = True
        output = []
        failed_validators = []
        for validator in VALIDATORS:
            val_status, val_output = validator.validate_fn(
                file_path, validator.config_path)
            if not val_status:
                status = False
                output.append(VALIDATOR_FMT(name=validator.name,
                                            u='-' * len(validator.name),
                                            out=val_output))
                failed_validators.append(validator.name)
        if not status:
            LOG.info(OUTPUT_FMT, file_path, ', '.join(failed_validators), '\n'.join(output))
    return status


# Validate all the files in the root_dir passed as argument
def validate_dir(root_dir):
    if root_dir in EXCLUDE_DIRECTORIES:
        return True

    status = True
    for root, dirs, files in os.walk(root_dir):  # pylint: disable=not-an-iterable
        # Remove excluded dirs from list
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in EXCLUDE_DIRECTORIES]
        for file_path in files:
            file_path = os.path.join(root, file_path)

            if not validate_file(file_path):
                status = False
    return status


# ==============================================
# VALIDATOR FUNCTION DEFINITIONS
# ==============================================

def check_pylint(file_path, config_path=None):
    options = [
        '--output-format=json',
        '--reports=yes',
    ]
    if config_path is not None:
        options.append('--rcfile=' + config_path)

    with settings(warn_only=True), quiet():
        res = local('pylint {} {}'.format(' '.join(options), file_path), capture=True)
    if res.stdout == '':
        assert res.return_code == 0, 'return_code={}, expected=0\n{}'.format(
            res.return_code, res.stderr)
        return True, None

    output = []
    errors = json.loads(res.stdout)
    for entry in errors:
        # Remove extra whitespace and hints
        msg = entry['message'].replace('^', '').replace('|', '')
        msg = re.sub(' +', ' ', msg)
        msg = msg.strip()
        output.append(format_message(os.path.basename(file_path), entry['line'],
                                     msg, entry['symbol']))
    output = ''.join(output)
    return res.return_code == 0, output


def check_pycodestyle(file_path, config_path=None):
    import pycodestyle

    # A custom reporter class for pycodestyle that checks for disabled errors
    # and formats the style report output.
    class CustomReporter(pycodestyle.StandardReport):
        def get_file_results(self):
            # Iterates through the lines of code that generated lint errors and
            # checks if the given error has been disabled for that line via an
            # inline comment (e.g., # pycodestyle: disable=E201,E226). Those
            # that have been disabled are not treated as errors.
            self._deferred_print.sort()
            results = []
            prev_line_num = -1
            prev_line_errs = []
            for line_number, _, code, text, _ in self._deferred_print:
                if prev_line_num == line_number:
                    err_codes = prev_line_errs
                else:
                    line = self.lines[line_number - 1]
                    m = PYCODESTYLE_COMMENT_PATTERN.search(line)
                    if m and m.group(0):
                        err_codes = [ec.strip() for ec in m.group(0).split('=')[1].split(',')]
                    else:
                        err_codes = []
                prev_line_num = line_number
                prev_line_errs = err_codes
                if code in err_codes:
                    # Error is disabled in source
                    continue

                results.append(format_message(os.path.basename(file_path),
                                              self.line_offset + line_number,
                                              text, code))
            return results, len(results) == 0
    # END CustomReporter class

    options = {} if config_path is None else {'config_file': config_path}
    style = pycodestyle.StyleGuide(quiet=True, **options)

    # Set the reporter option to our custom one
    style.options.reporter = CustomReporter
    style.init_report()
    report = style.check_files([file_path])
    results, status = report.get_file_results()
    output = None if status else ''.join(results)
    return status, output


def check_illegal_patterns(file_path, config_path=None):  # pylint: disable=unused-argument
    status = True
    line_num = 1
    output = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            for pattern_info in ILLEGAL_PATTERNS:
                if not line.startswith('#') and pattern_info[0].search(line):
                    output.append(format_message(filename=os.path.basename(file_path),
                                                 line=line_num,
                                                 message=pattern_info[1]))
                    status = False
            line_num += 1
    output = None if status else ''.join(output)
    return status, output


def check_header(file_path, config_file=None):  # pylint: disable=unused-argument
    status = True
    output = None
    with open(file_path, 'r') as f:
        file_contents = f.read()

    header_match = HEADER_PATTERN.search(file_contents)
    filename = os.path.basename(file_path)
    if header_match:
        if filename not in header_match.group(0):
            status = False
            output = format_message(filename=filename, line=2,
                                    message="Incorrect filename in header")

    else:
        status = False
        output = format_message(filename=filename, line=1,
                                message='Missing header')
    return status, output


# ==============================================
# VALIDATORS
# ==============================================

# Struct for storing validator metadata
Validator = namedtuple('Validator', 'name validate_fn modules config_path')

VALIDATORS = [
    # Runs pylint on python source
    Validator('check_pylint', check_pylint, ['pylint'],
              os.path.join(OTTERTUNE_DIR, "script/formatting/config/pylintrc")),

    # Runs pycodestyle on python source
    Validator('check_pycodestyle', check_pycodestyle, ['pycodestyle'],
              os.path.join(OTTERTUNE_DIR, "script/formatting/config/pycodestyle")),

    # Checks that the python source files do not use illegal patterns
    Validator('check_illegal_patterns', check_illegal_patterns, [], None),

    # Checks that the source files have headers
    Validator('check_header', check_header, [], None)
]


# ==============================================
# MAIN FUNCTION
# ==============================================

def main():
    parser = argparse.ArgumentParser(description="Validate OtterTune's source code")
    parser.add_argument('paths', metavar='PATH', type=str, nargs='*',
                        help='Files or directories to (recursively) validate')
    parser.add_argument('--staged-files', action='store_true',
                        help='Apply the selected action(s) to all staged files (git)')
    args = parser.parse_args()

    LOG.info('\nRunning source validators:\n%s\n',
             '\n'.join('  ' + v.name for v in VALIDATORS))
    for validator in VALIDATORS:
        if not validate_validator(validator.modules, validator.config_path):
            sys.exit(EXIT_FAILURE)

    if args.staged_files:
        targets = [os.path.abspath(os.path.join(OTTERTUNE_DIR, f))
                   for f in subprocess.check_output(["git", "diff", "--name-only", "HEAD",
                                                     "--cached", "--diff-filter=d"]).split()]
        if not targets:
            LOG.error("No staged files or not calling from a repository. Exiting...")
            sys.exit(EXIT_FAILURE)
    elif args.paths:
        targets = args.paths
    else:
        targets = DEFAULT_DIRS

    for target in targets:
        target = os.path.abspath(target)
        if os.path.isfile(target):
            LOG.debug("Scanning file: %s\n", target)
            status = validate_file(target)
        elif os.path.isdir(target):
            LOG.debug("Scanning directory: %s\n", target)
            status = validate_dir(target)
        else:
            LOG.error("%s isn't a file or directory", target)
            sys.exit(EXIT_FAILURE)

    if not status:
        LOG.info(SEPARATOR + '\n')
        LOG.info("Validation NOT successful\n")
        sys.exit(EXIT_FAILURE)

    LOG.info("Validation successful\n")
    sys.exit(EXIT_SUCCESS)


if __name__ == '__main__':
    main()

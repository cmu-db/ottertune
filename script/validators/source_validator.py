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
from fabric.api import lcd, local, settings, quiet

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
    OTTERTUNE_DIR
]

# Directories that should NOT be checked
EXCLUDE_DIRECTORIES = [
    # Django-generated directories
    os.path.join(OTTERTUNE_DIR, "server/website/website/migrations"),

    # Source code files from json.org
    os.path.join(OTTERTUNE_DIR, "client/controller/src/main/java/com/controller/util/json"),

    # Django settings
    os.path.join(OTTERTUNE_DIR, 'server/website/website/settings'),

    # Docker files
    os.path.join(OTTERTUNE_DIR, 'docker'),
]

# Files that should NOT be checked
EXCLUDE_FILES = [
    # Django-generated files
    os.path.join(OTTERTUNE_DIR, 'server/website/manage.py'),

    # Docker files
    os.path.join(OTTERTUNE_DIR, 'server/website/createadmin.py'),
]

CHECKSTYLE_JAR_PATH = os.path.join(OTTERTUNE_DIR,
                                   "client/controller/build/libs/checkstyle-8.8-all.jar")

# Regex patterns
PYCODESTYLE_COMMENT_PATTERN = re.compile(r'#\s*pycodestyle:\s*disable\s*=\s*[\w\,\s]+$')

PYTHON_ILLEGAL_PATTERNS = [
    (re.compile(r'^print[ (]'), "Do not use 'print'. Use the logging module instead.")
]

JAVA_ILLEGAL_PATTERNS = [
    (re.compile(r'^System.out.println'), "Do not use println. Use the logging module instead.")
]

PYTHON_HEADER_PATTERN = re.compile(r'#\n#.*\n#\n# Copyright.*\n#\n')
JAVA_HEADER_PATTERN = re.compile(r'/\*\n \*.*\n \*\n \* Copyright.*\n \*/\n\n')

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
    if file_path in EXCLUDE_FILES:
        return True
    if not file_path.endswith(".py") and not file_path.endswith(".java"):
        return True

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
    if not file_path.endswith(".py"):
        return True, None

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

    if not file_path.endswith(".py"):
        return True, None

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


def check_java_checkstyle(file_path, config_path=None):
    if not file_path.endswith(".java"):
        return True, None

    if not os.path.exists(CHECKSTYLE_JAR_PATH):
        with lcd(os.path.join(OTTERTUNE_DIR, "client/controller")):  # pylint: disable=not-context-manager
            local("gradle downloadJars")

    options = '' if config_path is None else '-c ' + config_path
    with quiet():
        res = local("java -jar {} {} {}".format(CHECKSTYLE_JAR_PATH, options, file_path),
                    capture=True)
    lines = res.stdout.split('\n')
    assert len(lines) >= 2 and lines[0] == "Starting audit..." and lines[-1] == "Audit done."
    if len(lines) == 2:
        return True, None
    output = []
    for line in lines[1:-1]:
        parts = line.strip().split(':')
        line_number = int(parts[1])
        text, code = parts[-1].rsplit('[', 1)
        text = text.strip()
        code = code[:-1]
        output.append(format_message(os.path.basename(file_path), line_number, text, code))
    output = ''.join(output)
    return False, output


def check_illegal_patterns(file_path, config_path=None):  # pylint: disable=unused-argument
    if file_path.endswith(".py"):
        illegal_patterns = PYTHON_ILLEGAL_PATTERNS
        comment = "#"
    elif file_path.endswith(".java"):
        illegal_patterns = JAVA_ILLEGAL_PATTERNS
        comment = "//"
    else:
        return True, None

    line_num = 1
    output = []
    status = True
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            for pattern_info in illegal_patterns:
                if not line.startswith(comment) and pattern_info[0].search(line):
                    output.append(format_message(filename=os.path.basename(file_path),
                                                 line=line_num,
                                                 message=pattern_info[1]))
                    status = False
            line_num += 1
    output = None if status else ''.join(output)
    return status, output


def check_header(file_path, config_file=None):  # pylint: disable=unused-argument
    if file_path.endswith(".py"):
        header_pattern = PYTHON_HEADER_PATTERN
    elif file_path.endswith(".java"):
        header_pattern = JAVA_HEADER_PATTERN
    else:
        return True, None

    status = True
    output = None
    with open(file_path, 'r') as f:
        file_contents = f.read()

    header_match = header_pattern.search(file_contents)
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

    # Runs checkstyle on the java source
    Validator("check_java_checkstyle", check_java_checkstyle, [],
              os.path.join(OTTERTUNE_DIR, "script/formatting/config/google_checks.xml")),

    # Checks that the python/java source files do not use illegal patterns
    Validator('check_illegal_patterns', check_illegal_patterns, [], None),

    # Checks that the python/java source files have headers
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

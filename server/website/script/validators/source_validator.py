#!/usr/bin/env python
# encoding: utf-8

# ==============================================
# SOURCE VALIDATOR
# ==============================================

# Adapted from the source validator used by Peloton.
# (see https://github.com/cmu-db/peloton/blob/master/script/validators/source_validator.py)

import argparse
import logging
import os
import re
import sys
import imp
import functools
from collections import namedtuple

EXIT_SUCCESS = 0
EXIT_FAILURE = -1


# ==============================================
# LOGGING CONFIGURATION
# ==============================================

LOG = logging.getLogger(__name__)
LOG_HANDLER = logging.StreamHandler()
LOG_FORMATTER = logging.Formatter(
    fmt='%(asctime)s [%(funcName)s:%(lineno)03d] %(levelname)-5s: %(message)s',
    datefmt='%H:%M:%S'
)
LOG_HANDLER.setFormatter(LOG_FORMATTER)
LOG.addHandler(LOG_HANDLER)
LOG.setLevel(logging.INFO)


# ==============================================
# CONFIGURATION
# ==============================================

# NOTE: absolute path to ottertune directory is calculated from current directory
# directory structure: ottertune/server/website/scripts/validators/<this_file>
# OTTERTUNE_DIR needs to be redefined if the directory structure is changed
CODE_SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
OTTERTUNE_DIR = os.path.abspath(functools.reduce(os.path.join,
                                                 [CODE_SOURCE_DIR,
                                                  os.path.pardir,
                                                  os.path.pardir,
                                                  os.path.pardir,
                                                  os.path.pardir]))

# Other directory paths used are relative to PELOTON_DIR
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

    # TODO (dvanaken): remove these and format files after merging #49
    os.path.join(OTTERTUNE_DIR, "server/website/script/controller_simulator/upload_data.py"),
    os.path.join(OTTERTUNE_DIR, "server/website/script/upload/upload.py"),
    os.path.join(OTTERTUNE_DIR, "server/website/script/upload/upload_batch.py"),

    # TODO (dvanaken): remove this and format files after merging parser tests
    os.path.join(OTTERTUNE_DIR, "server/website/website/utils.py"),

    # TODO (dvanaken): remove these and format files after merging #62 (parser checks) and
    # parser tests (for postgres.py only)
    os.path.join(OTTERTUNE_DIR, "server/website/website/parser/base.py"),
    os.path.join(OTTERTUNE_DIR, "server/website/website/parser/postgres.py"),
]

# Output format
THICK_LINE = '=' * 80 + '\n'
THIN_LINE = '-' * 80 + '\n'
OUTPUT_FORMAT = (THICK_LINE + "\n\033[1m %s: FAILED\n %s\033[0m\n" + THIN_LINE + "%s")


# ==============================================
# VALIDATOR CONFIGURATION
# ==============================================

# Namedtuple for storing regex pattern information
PatternInfo = namedtuple("PatternInfo", ['name', 'pattern', 'message'])

# check_pylint
PYLINT_FORMAT_PATTERNS = [
    PatternInfo('line_num', re.compile(r'(?<=:)\d+(?=:)'), None),
    PatternInfo('msg', re.compile(r'(?<=\) ).*$'), None),
    PatternInfo('msg_id', re.compile(r'(?<=\()[a-zA-z][0-9]+(?=\,)'), None),
    PatternInfo('symbol', re.compile(r'(?<=\, )\w[\w-]+(?=\,)'), None),
]
PYLINT_FORMAT = " - line {line_num}: {msg} ({msg_id}: {symbol})".format

# check_pycodestyle
PYCODESTYLE_COMMENT_PATTERN = re.compile(r'#\s*pycodestyle:\s*disable\s*=\s*[\w\,\s]+$')
PYCODESTYLE_FORMAT = " - line {line_num}:{offset} {msg} ({msg_id})".format

# check_illegal_patterns
ILLEGAL_PATTERNS = [
    PatternInfo('print', re.compile(r'^print[ (]'),
                'Print statements are not permitted. Use the logging module instead.'),
]
ILLEGAL_FORMAT = " - line {line_num}: {msg}".format


# ==============================================
# VALIDATOR FUNCTION DEFINITIONS
# ==============================================

def check_pylint(file_path, config_path=None):
    from pylint import epylint as lint

    options = []
    if config_path is not None:
        options.append('--rcfile=' + config_path)
    options = ' '.join(options)
    stdout_file, stderr_file = lint.py_run(file_path + ' ' + options, return_std=True)
    stdout, stderr = stdout_file.read(), stderr_file.read()
    status = stdout == '' and stderr == ''
    if status:
        output = None
    else:
        output = []
        if stdout != '':
            # Fix pylint's output format (stdout)
            for line in stdout.strip().split('\n')[1:]:
                line = line.strip()
                if line == '':
                    continue
                format_vars = {}
                for pattern_info in PYLINT_FORMAT_PATTERNS:
                    m = pattern_info.pattern.search(line)
                    format_vars[pattern_info.name] = '' if m is None else m.group()
                output.append(PYLINT_FORMAT(**format_vars))
            output = '\n'.join(output) + '\n'
        if stderr != '':
            output += stderr + '\n'
    return status, output


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
            for line_number, offset, code, text, _ in self._deferred_print:
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

                # Format the output of the style checker
                results.append(PYCODESTYLE_FORMAT(
                    line_num=self.line_offset + line_number,
                    offset=offset + 1, msg=text, msg_id=code))
            return results, len(results) == 0
    # END CustomReporter class

    options = {} if config_path is None else {'config_file': config_path}
    style = pycodestyle.StyleGuide(quiet=True, **options)

    # Update the reporter to be our custom one
    style.options.reporter = CustomReporter
    style.init_report()
    report = style.check_files([file_path])
    results, status = report.get_file_results()
    output = None if status else '\n'.join(results) + '\n'
    return status, output


def check_illegal_patterns(file_path, config_path=None):  # pylint: disable=unused-argument
    with open(file_path, 'r') as f:
        status = True
        line_num = 1
        output = []
        for line in f:
            line = line.strip()
            for pattern_info in ILLEGAL_PATTERNS:
                if not line.startswith('#') and pattern_info.pattern.search(line):
                    output.append(ILLEGAL_FORMAT(line_num=line_num, msg=pattern_info.message))
                    status = False
            line_num += 1
        output = None if status else '\n'.join(output) + '\n'
        return status, output


# ==============================================
# UTILITY FUNCTION DEFINITIONS
# ==============================================

def validate_validator(modules, config_path):
    status = True

    # Check that the required modules are installed
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
        status = True
        outputs = []
        for validator in VALIDATORS:
            val_status, val_output = validator.validate_fn(
                file_path, validator.config_path)
            if not val_status:
                status = False
                outputs.append((validator.name, val_output))
        if len(outputs) > 0:
            for name, output in outputs:
                LOG.info(OUTPUT_FORMAT, name, file_path, output)
            LOG.info(THICK_LINE)
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
# VALIDATORS
# ==============================================

# Struct for storing validator metadata
Validator = namedtuple('Validator', 'name validate_fn modules config_path')

VALIDATORS = [
    # Runs pylint on python source
    Validator('check_pylint', check_pylint, ['pylint'],
              os.path.join(OTTERTUNE_DIR, "server/.pylintrc")),

    # Runs pycodestyle on python source
    Validator('check_pycodestyle', check_pycodestyle, ['pycodestyle'],
              os.path.join(OTTERTUNE_DIR, "server/.pycodestyle")),

    # Checks that the python source files do not use illegal patterns
    Validator('check_illegal_patterns', check_illegal_patterns, [], None),
]


# ==============================================
# MAIN FUNCTION
# ==============================================

def main():
    parser = argparse.ArgumentParser(description="Validate OtterTune's source code")
    parser.add_argument("-f", "--files", nargs='*', help="A list of files to validate")
    args = parser.parse_args()

    LOG.info('Enabled source validators:\n%s\n', '\n'.join('  ' + v.name for v in VALIDATORS))
    for validator in VALIDATORS:
        if not validate_validator(validator.modules, validator.config_path):
            sys.exit(EXIT_FAILURE)

    LOG.info("Running source validators...")
    if args.files:
        # Validate just the provided files.
        for file_path in args.files:
            file_path = os.path.abspath(file_path.lower())

            # Fail if the file DNE or isn't really a file
            if not os.path.isfile(file_path):
                LOG.info("%s isn't a file", file_path)
                sys.exit(EXIT_FAILURE)

            # Looks good, let's validate
            LOG.info("Scanning file : %s\n", file_path)
            status = validate_file(file_path)
            if not status:
                LOG.info("Validation NOT successful")
                sys.exit(EXIT_FAILURE)

    else:
        # Validate all files in server directory
        for dir_path in DEFAULT_DIRS:
            LOG.info("Scanning directory : %s\n", dir_path)

            status = validate_dir(dir_path)
            if not status:
                LOG.info("Validation NOT successful")
                sys.exit(EXIT_FAILURE)

    LOG.info("Validation successful")
    sys.exit(EXIT_SUCCESS)


if __name__ == '__main__':
    main()

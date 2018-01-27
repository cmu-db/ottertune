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
    datefmt='%m-%d-%Y %H:%M:%S'
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

# Directories that should not be checked
EXCLUDE_DIRECTORIES = [
    os.path.join(OTTERTUNE_DIR, "server/website/tests"),
    os.path.join(OTTERTUNE_DIR, "server/website/website/migrations"),
]

# Files that should not be checked
EXCLUDE_FILES = [
    # Django files
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


# ==============================================
# VALIDATOR FUNCTION DEFINITIONS
# ==============================================

def check_pylint(file_path, config_path=None):
    from pylint import epylint as lint

    LOG.debug("Running pylint on " + file_path)
    options = []
    if config_path is not None:
        options.append('--rcfile=' + config_path)
    return_code = lint.lint(file_path, options=options)
    return return_code == 0


def check_pycodestyle(file_path, config_path=None):
    import pycodestyle

    # We set pycodestyle to ignore line length (E501) because pylint
    # already checks this so we can avoid having to create a custom
    # config file for pycodestyle.
    LOG.debug("Running pycodestyle on " + file_path)
    options = {}
    if config_path is not None:
        options['config_file'] = config_path
    style = pycodestyle.StyleGuide(**options)
    result = style.check_files([file_path])
    return result.total_errors == 0


# ==============================================
# UTILITY FUNCTION DEFINITIONS
# ==============================================

def validate_validator(module_name, config_path):
    status = True

    # Check that the module is installed
    try:
        imp.find_module(module_name)
    except ImportError:
        LOG.error("Cannot find " + module_name + " module")
        status = False

    # Check that the config file exists if assigned
    if config_path is not None and not os.path.isfile(config_path):
        LOG.error("Cannot find " + module_name + " config file: " + config_path)
        status = False
    return status


# Validate the file passed as argument
def validate_file(file_path):
    if file_path in EXCLUDE_FILES or not file_path.endswith(".py"):
        status = True
    else:
        status = True
        for validator in VALIDATORS:
            if not validator.validate_fn(file_path, validator.config_path):
                status = False
    return status


# Validate all the files in the root_dir passed as argument
def validate_dir(root_dir):
    LOG.info("OtterTune root : " + root_dir)
    if root_dir in EXCLUDE_DIRECTORIES:
        return True

    status = True
    for root, dirs, files in os.walk(root_dir):  # pylint: disable=not-an-iterable
        # Remove any excluded dirs
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in EXCLUDE_DIRECTORIES]
        for file_path in files:
            file_path = os.path.join(root, file_path)

            if not validate_file(file_path):
                status = False
    return status


# ==============================================
# VALIDATORS
# ==============================================

Validator = namedtuple('Validator', 'module_name config_path validate_fn')
VALIDATORS = [
    # Runs pylint on python source
    Validator('pylint', os.path.join(OTTERTUNE_DIR, "server/.pylintrc"), check_pylint),

    # Runs pycodestyle on python source
    Validator('pycodestyle', os.path.join(OTTERTUNE_DIR, "server/.pycodestyle"), check_pycodestyle),
]


# ==============================================
#                 Main Function
# ==============================================

def main():
    parser = argparse.ArgumentParser(description="Validate OtterTune's source code")
    parser.add_argument("-f", "--files", nargs='*', help="A list of files to validate")
    args = parser.parse_args()

    LOG.info('Checking validators...')
    for validator in VALIDATORS:
        if not validate_validator(validator.module_name, validator.config_path):
            sys.exit(EXIT_FAILURE)

    LOG.info("Running source validators...")
    if args.files:
        # Validate just the provided files.
        for filepath in args.files:
            filepath = os.path.abspath(filepath.lower())

            # Fail if the file isn't really a file
            if not os.path.isfile(filepath):
                LOG.info("ERROR: " + filepath + " isn't a file")
                sys.exit(EXIT_FAILURE)

            # Looks good, let's validate
            LOG.info("Scanning file : " + filepath)
            status = validate_file(filepath)
            if not status:
                LOG.info("Validation NOT successful")
                sys.exit(EXIT_FAILURE)

    else:
        # Validate all files in server directory
        for dir_path in DEFAULT_DIRS:
            LOG.info("Scanning directory : " + dir_path)

            status = validate_dir(dir_path)
            if not status:
                LOG.info("Validation NOT successful")
                sys.exit(EXIT_FAILURE)

    LOG.info("Validation successful")
    sys.exit(EXIT_SUCCESS)


if __name__ == '__main__':
    main()

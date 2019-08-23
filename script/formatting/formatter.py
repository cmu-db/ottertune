#
# OtterTune - formatter.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import argparse
import functools
import logging
import os
import re
import subprocess
import sys

import autopep8

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

# NOTE: the absolute path to ottertune directory is calculated from current
# directory structure: ottertune/server/website/scripts/validators/<this_file>
# OTTERTUNE_DIR needs to be redefined if the directory structure is changed.
CODE_SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
OTTERTUNE_DIR = os.path.abspath(functools.reduce(os.path.join,
                                                 [CODE_SOURCE_DIR,
                                                  os.path.pardir,
                                                  os.path.pardir]))

JAVA_JAR_PATH = os.path.join(
    OTTERTUNE_DIR, 'controller/build/libs/google-java-format-1.5-all-deps.jar')

# ==============================================
# FILE HEADER FORMATS
# ==============================================

PYTHON_HEADER_FORMAT = (
    "#\n"
    "# OtterTune - {filename}\n"
    "#\n"
    "# Copyright (c) 2017-18, Carnegie Mellon University Database Group\n"
    "#\n"
).format

# Regex for updating old headers
PYTHON_HEADER_REGEX = re.compile(r'#\n#.*\n#\n# Copyright.*\n#\n')

JAVA_HEADER_FORMAT = (
    "/*\n"
    " * OtterTune - {filename}\n"
    " *\n"
    " * Copyright (c) 2017-18, Carnegie Mellon University Database Group\n"
    " */\n\n"
).format

JAVA_HEADER_REGEX = re.compile(r'/\*\n \*.*\n \*\n \* Copyright.*\n \*/\n\n')

# ==============================================
# UTILITY FUNCTION DEFINITIONS
# ==============================================


def format_file(file_path, update_header, format_code):
    if file_path.endswith(".py"):
        format_python_file(file_path, update_header, format_code)
    elif file_path.endswith(".java"):
        format_java_file(file_path, update_header, format_code)


def update_file_header(file_contents, file_name, header_format, header_regex):
    new_header = header_format(filename=os.path.basename(file_name))
    header_match = header_regex.search(file_contents)
    if header_match:
        # Replace the old header with the new one
        old_header = header_match.group()
        file_contents = file_contents.replace(old_header, new_header)
    else:
        # Add new header
        file_contents = new_header + file_contents
    return file_contents


def format_java_file(file_path, update_header, format_code):
    if not file_path.endswith(".java"):
        return

    if update_header:
        with open(file_path, 'r') as f:
            file_contents = f.read()
        file_contents = update_file_header(file_contents,
                                           os.path.basename(file_path),
                                           JAVA_HEADER_FORMAT,
                                           JAVA_HEADER_REGEX)
        with open(file_path, 'w') as f:
            f.write(file_contents)

    if format_code:
        if not os.path.exists(JAVA_JAR_PATH):
            controller_dir = os.path.join(OTTERTUNE_DIR, 'controller')
            subprocess.check_output(["gradle", "downloadJars"], cwd=controller_dir)
        subprocess.check_output(["java", "-jar", JAVA_JAR_PATH, "-r", file_path])


def format_python_file(file_path, update_header, format_code):
    if not file_path.endswith(".py"):
        return

    with open(file_path, 'r') as f:
        file_contents = f.read()

    if update_header:
        file_contents = update_file_header(file_contents,
                                           os.path.basename(file_path),
                                           PYTHON_HEADER_FORMAT,
                                           PYTHON_HEADER_REGEX)

    if format_code:
        # Use the autopep8 module to format the source code. autopep8 uses
        # pycodestyle to detect the style errors it should fix and thus it
        # should fix all (or most) of them, however, it does not use pylint
        # so it may not fix all of its reported errors.
        options = {"max_line_length": 100}
        file_contents = autopep8.fix_code(file_contents, options=options)

    with open(file_path, 'w') as f:
        f.write(file_contents)


# Format all the files in the dir passed as argument
def format_dir(dir_path, update_header, format_code):
    for subdir, _, files in os.walk(dir_path):  # pylint: disable=not-an-iterable
        for file_path in files:
            file_path = subdir + os.path.sep + file_path
            format_file(file_path, update_header, format_code)


def main():
    parser = argparse.ArgumentParser(description='Formats python source files in place')
    parser.add_argument('--no-update-header', action='store_true',
                        help='Do not update the source file headers')
    parser.add_argument('--no-format-code', action='store_true',
                        help='Do not format the source files use autopep8')
    parser.add_argument('--staged-files', action='store_true',
                        help='Apply the selected action(s) to all staged files (git)')
    parser.add_argument('paths', metavar='PATH', type=str, nargs='*',
                        help='Files or directories to (recursively) apply the actions to')
    args = parser.parse_args()

    if args.no_update_header and args.no_format_code:
        LOG.info("No actions to perform (both --no-update-header and "
                 "--no-format-code given). Exiting...")
        sys.exit(EXIT_FAILURE)
    elif args.staged_files:
        targets = [os.path.abspath(os.path.join(OTTERTUNE_DIR, f))
                   for f in subprocess.check_output(["git", "diff",
                                                     "--name-only", "HEAD",
                                                     "--cached",
                                                     "--diff-filter=d"]).split()]
        if not targets:
            LOG.error("No staged files or not calling from a repository. Exiting...")
            sys.exit(EXIT_FAILURE)
    elif not args.paths:
        LOG.error("No files or directories given. Exiting...")
        sys.exit(EXIT_FAILURE)
    else:
        targets = args.paths

    for x in targets:
        if os.path.isfile(x):
            LOG.info("Scanning file: " + x)
            format_file(x, not args.no_update_header, not args.no_format_code)
        elif os.path.isdir(x):
            LOG.info("Scanning directory: " + x)
            format_dir(x, not args.no_update_header, not args.no_format_code)
        else:
            LOG.error("%s isn't a file or directory", x)
            sys.exit(EXIT_FAILURE)


if __name__ == '__main__':
    main()

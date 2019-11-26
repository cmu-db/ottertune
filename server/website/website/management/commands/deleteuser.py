#
# OtterTune - deleteuser.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

from website.utils import delete_user  # pylint: disable=no-name-in-module,import-error


class Command(BaseCommand):
    help = 'Delete an existing user.'

    def add_arguments(self, parser):
        parser.add_argument(
            'username',
            metavar='USERNAME',
            # required=True,
            help='Specifies the login of the user to delete.')

    def handle(self, *args, **options):
        username = options['username']
        _, deleted = delete_user(username)
        if deleted:
            self.stdout.write(self.style.SUCCESS(
                "Successfully deleted user '{}'.".format(username)))
        else:
            self.stdout.write(self.style.NOTICE(
                "ERROR: User '{}' does not exist.".format(username)))

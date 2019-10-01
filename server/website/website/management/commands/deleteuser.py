#
# OtterTune - deleteuser.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


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
        try:
            user = User.objects.get(username=username)
            user.delete()
            self.stdout.write(self.style.SUCCESS(
                "Successfully deleted user '{}'.".format(username)))
        except User.DoesNotExist:
            self.stdout.write(self.style.NOTICE(
                "ERROR: User '{}' does not exist.".format(username)))

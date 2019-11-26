#
# OtterTune - createuser.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

from website.utils import create_user  # pylint: disable=no-name-in-module,import-error


class Command(BaseCommand):
    help = 'Create a new user.'

    def add_arguments(self, parser):
        parser.add_argument(
            'username',
            metavar='USERNAME',
            help='Specifies the login for the user.')
        parser.add_argument(
            'password',
            metavar='PASSWORD',
            help='Specifies the password for the user.')
        parser.add_argument(
            '--email',
            metavar='EMAIL',
            default=None,
            help='Specifies the email for the user.')
        parser.add_argument(
            '--superuser',
            action='store_true',
            help='Creates a superuser.')

    def handle(self, *args, **options):
        username = options['username']
        password = options['password']
        email = options['email']
        superuser = options['superuser']

        _, created = create_user(username, password, email, superuser)

        if created:
            self.stdout.write(self.style.SUCCESS("Successfully created {} '{}'{}.".format(
                'superuser' if superuser else 'user', username,
                " ('{}')".format(email) if email else '')))
        else:
            self.stdout.write(self.style.NOTICE(
                "ERROR: User '{}' already exists.".format(username)))

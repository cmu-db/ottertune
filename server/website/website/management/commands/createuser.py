#
# OtterTune - createuser.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


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
        if User.objects.filter(username=username).exists():
            self.stdout.write(self.style.NOTICE(
                "ERROR: User '{}' already exists.".format(username)))
        else:
            password = options['password']
            email = options['email']
            superuser = options['superuser']

            if superuser:
                email = email or '{}@noemail.com'.format(username)
                create_user = User.objects.create_superuser
            else:
                create_user = User.objects.create_user

            create_user(username=username, password=password, email=email)

            self.stdout.write(self.style.SUCCESS("Successfully created {} '{}'{}.".format(
                'superuser' if superuser else 'user', username,
                " ('{}')".format(email) if email else '')))

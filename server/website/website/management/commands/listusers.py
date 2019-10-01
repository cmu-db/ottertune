#
# OtterTune - listusers.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'List all users.'
    default_fields = ('username', 'email', 'is_superuser')
    item_fmt = '{{{i}: <{w}}}'.format
    col_space = 3

    def add_arguments(self, parser):
        parser.add_argument(
            'fields',
            nargs='*',
            default='DEFAULT',
            choices=[f.name for f in User._meta.get_fields()] + ['DEFAULT'],
            metavar='FIELDS',
            help='Fields from the User model to display. (default: {})'.format(
                list(self.default_fields)))

    def handle(self, *args, **options):
        fields = options['fields']
        if fields == 'DEFAULT':
            fields = self.default_fields

        users = User.objects.values_list(*fields)
        self.stdout.write(self.style.NOTICE(
            '\nFound {} existing users.\n'.format(len(users))))
        if users:
            fmt = ''
            for i, field in enumerate(fields):
                w = max(len(field), max(len(str(u[i])) for u in users)) + self.col_space
                fmt += self.item_fmt(i=i, w=w)
            fmt = (fmt + '\n').format
            h = fmt(*fields)
            out = h + ('-' * (len(h) + 1)) + '\n'
            for user_info in users:
                out += fmt(*(str(ui) for ui in user_info))
            out += '\n'

            self.stdout.write(out)

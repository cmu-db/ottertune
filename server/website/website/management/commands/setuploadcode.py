#
# OtterTune - setuploadcode.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.core.management.base import BaseCommand

from website.models import Session
from website.utils import MediaUtil


class Command(BaseCommand):
    help = 'Set the upload code for a session.'

    def add_arguments(self, parser):
        parser.add_argument(
            'username',
            metavar='USERNAME',
            help='Specifies the username of the session owner.')
        parser.add_argument(
            'projectname',
            metavar='PROJECTNAME',
            help='Specifies the name of the project that the session belongs to.')
        parser.add_argument(
            'sessionname',
            metavar='SESSIONNAME',
            help='Specifies the name of the session.')
        parser.add_argument(
            '--uploadcode',
            metavar='UPLOADCODE',
            default=None,
            help='Specifies the value to set the upload code to.')

    def handle(self, *args, **options):
        username = options['username']
        projectname = options['projectname']
        sessionname = options['sessionname']
        session = Session.objects.filter(user__username=username,
                                         project__name=projectname,
                                         name=sessionname).first()
        if session:
            upload_code = options['uploadcode'] or MediaUtil.upload_code_generator()
            session.upload_code = upload_code
            session.save()
            self.stdout.write(self.style.NOTICE(upload_code))
        else:
            self.stdout.write(self.style.NOTICE(
                "ERROR: Session '{}' for user '{}' under project '{}' does not exist.".format(
                    sessionname, username, projectname)))

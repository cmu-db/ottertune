#
# OtterTune - cleardblog.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.core.management.base import BaseCommand
from django_db_logger.models import StatusLog


class Command(BaseCommand):
    help = 'Clear all log entries from the django_db_logger table.'

    def handle(self, *args, **options):
        StatusLog.objects.all().delete()
        self.stdout.write(self.style.SUCCESS(
            "Successfully cleared the django_db_logger table."))

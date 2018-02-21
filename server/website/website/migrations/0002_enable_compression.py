# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations

TABLES_TO_COMPRESS = [
    "website_backupdata",
    "website_knobdata",
    "website_metricdata",
    "website_pipelinedata",
]


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0001_initial'),
    ]

    operations = [
        migrations.RunSQL(["ALTER TABLE " + table_name + " COMPRESSION='zlib';",
                           "OPTIMIZE TABLE " + table_name + ";"],
                          ["ALTER TABLE " + table_name + " COMPRESSION='none';",
                           "OPTIMIZE TABLE " + table_name + ";"])
                for table_name in TABLES_TO_COMPRESS
    ]

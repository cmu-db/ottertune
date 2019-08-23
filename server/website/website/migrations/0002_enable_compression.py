# -*- coding: utf-8 -*-

import logging

from django.conf import settings
#from django.core.exceptions import ProgrammingError 
from django.db import connection, migrations, ProgrammingError

LOG = logging.getLogger(__name__)

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

    try:
        LOG.info('***** VENDOR: %s', connection.vendor)
        if connection.vendor == 'mysql':

            version = (0, 0, 0)
            with connection.cursor() as cursor:
                cursor.execute('SELECT VERSION()')
                version = cursor.fetchone()[0]

            version = version.split('-')[0]
            version = version.split('.')
            version = tuple(int(v) for v in version)
            LOG.info('***** DB VERSION: %s', version)

            if version >= (5, 7, 0):
                operations = [
                    migrations.RunSQL(["ALTER TABLE " + table_name + " COMPRESSION='zlib';",
                                       "OPTIMIZE TABLE " + table_name + ";"],
                                      ["ALTER TABLE " + table_name + " COMPRESSION='none';",
                                       "OPTIMIZE TABLE " + table_name + ";"])
                            for table_name in TABLES_TO_COMPRESS
                ]
                LOG.info('***** DONE ENABLING COMPRESSION')

            else:
                operations = []
                LOG.info('***** COMPRESSION NOT SUPPORTED: %s < (5, 7, 0)', version)

        else:
            LOG.info('***** DB COMPRESSION NOT SUPPORTED: %s', connection.vendor)

    except ProgrammingError as err:
        LOG.warning("Error applying migration '0002_enable_compression'... Skipping")
        operations = []


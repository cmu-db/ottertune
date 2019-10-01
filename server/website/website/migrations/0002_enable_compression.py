# -*- coding: utf-8 -*-

import logging

from django.db import migrations, ProgrammingError

LOG = logging.getLogger(__name__)

TABLES_TO_COMPRESS = [
    "website_backupdata",
    "website_knobdata",
    "website_metricdata",
    "website_pipelinedata",
]

MYSQL_MIN_VERSION = (5, 7, 0)

ALTER_SQL = "ALTER TABLE %s COMPRESSION='%s'"
OPTIMIZE_SQL = "OPTIMIZE TABLE %s"


def compression_supported(schema_editor):
    supported = False
    dbms = schema_editor.connection.vendor

    if dbms == 'mysql':
        with schema_editor.connection.cursor() as cursor:
            cursor.execute('SELECT VERSION()')
            res = cursor.fetchone()[0]

        version_str = res.split('-')[0]
        version = tuple(int(v) for v in version_str.split('.'))
        assert len(version) == len(MYSQL_MIN_VERSION), \
            'MySQL - current version: {}, min version: {}'.format(version, MYSQL_MIN_VERSION)

        if version >= MYSQL_MIN_VERSION:
            supported = True
            LOG.debug("%s %s: table compression supported.", dbms.upper(), version_str)
        else:
            LOG.debug("%s %s: table compression NOT supported.", dbms.upper(), version_str)
    else:
        LOG.debug("%s: table compression NOT supported.", dbms.upper())

    return supported


def enable_compression(apps, schema_editor):
    # try:
    if compression_supported(schema_editor):
        for table in TABLES_TO_COMPRESS:
            schema_editor.execute(ALTER_SQL % (table, 'zlib'))
            schema_editor.execute(OPTIMIZE_SQL % table)

    # except ProgrammingError:
    #     LOG.warning("Error applying forward migration '0002_enable_compression'... Skipping.")


def disable_compression(apps, schema_editor):
    try:
        if compression_supported(schema_editor):
            for table in TABLES_TO_COMPRESS:
                schema_editor.execute(ALTER_SQL % (table, 'none'))
                schema_editor.execute(OPTIMIZE_SQL % table)

    except ProgrammingError:
        LOG.warning("Error applying reverse migration '0002_enable_compression'... Skipping.")


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0001_initial'),
    ]

    operations = [migrations.RunPython(enable_compression, disable_compression)]



#
# OtterTune - oracle.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from .base import BaseParser
from website.models import DBMSCatalog
from website.types import DBMSType


class OracleParser(BaseParser):

    def __init__(self, dbms_id):
        super(OracleParser, self).__init__(dbms_id)
        self.valid_true_val = ["TRUE", "true", "yes", 1]
        self.valid_false_val = ["FALSE", "false", "no", 0]

    ORACLE_BASE_KNOBS = {
    }

    @property
    def base_configuration_settings(self):
        return dict(self.ORACLE_BASE_KNOBS)

    @property
    def knob_configuration_filename(self):
        return 'initorcldb.ora'

    @property
    def transactions_counter(self):
        return 'global.user commits'

    @property
    def latency_timer(self):
        return 'global.user commits'

    def parse_version_string(self, version_string):
        return version_string


class Oracle19Parser(OracleParser):

    def __init__(self):
        dbms = DBMSCatalog.objects.get(
            type=DBMSType.ORACLE, version='19.0.0.0.0')
        super(Oracle19Parser, self).__init__(dbms.pk)

#
# OtterTune - oracle.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from .base import BaseParser


class OracleParser(BaseParser):

    def __init__(self, dbms_obj):
        super().__init__(dbms_obj)
        self.true_value = 'TRUE'
        self.false_value = 'FALSE'

    @property
    def transactions_counter(self):
        return 'global.user commits'

    @property
    def latency_timer(self):
        raise NotImplementedError()

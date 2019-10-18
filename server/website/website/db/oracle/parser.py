#
# OtterTune - parser.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from ..base.parser import BaseParser


class OracleParser(BaseParser):

    def __init__(self, dbms_obj):
        super().__init__(dbms_obj)
        self.true_value = 'TRUE'
        self.false_value = 'FALSE'
        self.bytes_system = (
            (1024 ** 4, 'T'),
            (1024 ** 3, 'G'),
            (1024 ** 2, 'M'),
            (1024 ** 1, 'k'),
        )
        self.min_bytes_unit = 'k'

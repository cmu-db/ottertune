#
# OtterTune - parser.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from website.types import KnobUnitType
from website.utils import ConversionUtil
from ..base.parser import BaseParser  # pylint: disable=relative-beyond-top-level


# pylint: disable=no-self-use
class MysqlParser(BaseParser):

    def __init__(self, dbms_obj):
        super().__init__(dbms_obj)
        self.bytes_system = (
            (1024 ** 4, 'T'),
            (1024 ** 3, 'G'),
            (1024 ** 2, 'M'),
            (1024 ** 1, 'k'),
        )
        self.time_system = None
        self.min_bytes_unit = 'k'
        self.valid_true_val = ("on", "true", "yes", '1', 'enabled')
        self.valid_false_val = ("off", "false", "no", '0', 'disabled')

    def convert_integer(self, int_value, metadata):
        # Collected knobs/metrics do not show unit, convert to int directly
        if len(str(int_value)) == 0:
            # The value collected from the database is empty
            return 0
        try:
            try:
                converted = int(int_value)
            except ValueError:
                converted = int(float(int_value))

        except ValueError:
            raise Exception('Invalid integer format for {}: {}'.format(
                metadata.name, int_value))
        return converted

    def format_integer(self, int_value, metadata):
        int_value = int(round(int_value))
        if int_value > 0 and metadata.unit == KnobUnitType.BYTES:
            int_value = ConversionUtil.get_human_readable2(
                int_value, self.bytes_system, self.min_bytes_unit)
        return int_value

    def parse_version_string(self, version_string):
        s = version_string.split('.')[0] + '.' + version_string.split('.')[1]
        return s

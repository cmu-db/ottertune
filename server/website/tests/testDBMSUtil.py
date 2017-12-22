
from django.test import TestCase
from website.parser.postgres import PostgresParser
from website.types import BooleanType, VarType, KnobUnitType, LabelStyleType

class BaseParserTest(TestCase):
    def testConverts(self):
        testDBMS = PostgresParser(2)

        # Convert Bool
        self.assertEqual(testDBMS.convert_bool('on', []), BooleanType.TRUE)
        self.assertEqual(testDBMS.convert_bool('ON', []), BooleanType.TRUE)
        self.assertEqual(testDBMS.convert_bool('OFF', []), BooleanType.FALSE)
        self.assertEqual(testDBMS.convert_bool('off', []), BooleanType.FALSE)
        self.assertEqual(testDBMS.convert_bool('apavlo', []), BooleanType.FALSE)

        # Convert Enum
        EnumParam = VarType()
        EnumParam.enumvals = 'apple,oranges,cake'
        EnumParam.name = "Test"

        self.assertEqual(testDBMS.convert_enum('apple', EnumParam), 0)
        self.assertEqual(testDBMS.convert_enum('oranges', EnumParam), 1)
        self.assertEqual(testDBMS.convert_enum('cake', EnumParam), 2)

        with self.assertRaises(Exception):
            testDBMS.convert_enum('jackyl', EnumParam)

        # Convert Integer

        IntParam = VarType()

        self.assertEqual(testDBMS.convert_integer('42', IntParam), 42)
        self.assertEqual(testDBMS.convert_integer('-1', IntParam), -1)
        self.assertEqual(testDBMS.convert_integer('0', IntParam), 0)
        self.assertEqual(testDBMS.convert_integer('1', IntParam), 1)

        self.assertEqual(testDBMS.convert_integer('42.0'), 42)
        self.assertEqual(testDBMS.convert_integer('42.5'), 42)
        self.assertEqual(testDBMS.convert_integer('42.7'), 42)

        with self.assertRaises(Exception):
            testDBMS.convert_integer('notInt', IntParam)

        # Convert Real

        RealParam = VarType()

        self.assertEqual(testDBMS.convert_real('42.0', RealParam), 42.0)
        self.assertEqual(testDBMS.convert_real('42.2', RealParam), 42.2)
        self.assertEqual(testDBMS.convert_real('42.5', RealParam), 42.5)
        self.assertEqual(testDBMS.convert_real('42.7', RealParam), 42.7)

        self.assertEqual(testDBMS.convert_real('-1', RealParam), -1.0)
        self.assertEqual(testDBMS.convert_real('0', RealParam), 0.0)
        self.assertEqual(testDBMS.convert_real('1', RealParam), 1.0)

        with self.assertRaises(Exception):
            testDBMS.convert_real('notReal', RealParam)

        # Convert DBMS Knobs

        # Convert DBMS Metrics

    def testExtractValidKeys(self):
        pass

    def testParses(self):
        # Parse Helper

        # Parse DBMS Config

        # Parse DBMS Metrics

        pass

    def testCalculateChangeInMetrics(self):
        pass

    def testCreateConfig(self):
        pass

    def testGetNondefaultSettings(self):
        pass

    def testFormats(self):
        # Format Bool

        # Format Enum

        # Format Integer

        # Format Real

        # Format DBMS Params

        # Format DBMS Metrics
        pass

    def testFilters(self):
        # Filter Numeric Metrics

        # Filter Tunable Parameters

        pass


class PostgresParserTest(TestCase):

    def testProperties(self):
        testDBMS = PostgresParser(2)
        baseConfig = testDBMS.base_configuration_settings
        baseConfigSet = set(baseConfig)
        self.assertTrue('global.data_directory' in baseConfigSet)
        self.assertTrue('global.hba_file' in baseConfigSet)
        self.assertTrue('global.ident_file' in baseConfigSet)
        self.assertTrue('global.external_pid_file' in baseConfigSet)
        self.assertTrue('global.listen_addresses' in baseConfigSet)
        self.assertTrue('global.port' in baseConfigSet)
        self.assertTrue('global.max_connections' in baseConfigSet)
        self.assertTrue('global.unix_socket_directories' in baseConfigSet)
        self.assertTrue('global.log_line_prefix' in baseConfigSet)
        self.assertTrue('global.track_counts' in baseConfigSet)
        self.assertTrue('global.track_io_timing' in baseConfigSet)
        self.assertTrue('global.autovacuum' in baseConfigSet)
        self.assertTrue('global.default_text_search_config' in baseConfigSet)

        self.assertEqual(testDBMS.knob_configuration_filename, 'postgresql.conf')
        self.assertEqual(testDBMS.transactions_counter, 'pg_stat_database.xact_commit')

    def testParseVersionString(self):
        testDBMS = PostgresParser(2)

        self.assertTrue(testDBMS.parse_version_string("9.6.1"), "9.6")
        self.assertTrue(testDBMS.parse_version_string("9.6.3"), "9.6")
        self.assertTrue(testDBMS.parse_version_string("10.2.1"), "10.2")
        self.assertTrue(testDBMS.parse_version_string("0.0.0"), "0.0")

        with self.assertRaises(Exception):
            testDBMS.parse_version_string("postgres")

        with self.assertRaises(Exception):
            testDBMS.parse_version_string("-1.0")



    def testConverts(self):
        testDBMS = PostgresParser(2)

        # Convert Integer
        KnobUnitBytes = KnobUnitType()
        KnobUnitBytes.unit = 1
        KnobUnitTime = KnobUnitType()
        KnobUnitTime.unit = 2
        KnobUnitOther = KnobUnitType()
        KnobUnitOther.unit = 3


        self.assertEqual(testDBMS.convert_integer('5', KnobUnitOther), 5)
        self.assertEqual(testDBMS.convert_integer('0', KnobUnitOther), 0)
        self.assertEqual(testDBMS.convert_integer('0.0', KnobUnitOther), 0)
        self.assertEqual(testDBMS.convert_integer('0.5', KnobUnitOther), 0)
        self.assertEqual(testDBMS.convert_integer('-0.5', KnobUnitOther), 0)
        self.assertEqual(testDBMS.convert_integer('-20', KnobUnitOther), -20)

        self.assertEqual(testDBMS.convert_integer('5kB', KnobUnitBytes), 5 * 1024)
        self.assertEqual(testDBMS.convert_integer('4MB', KnobUnitBytes), 4 * 1024 ** 2)

        self.assertEqual(testDBMS.convert_integer('1d', KnobUnitTime), 86400000)
        self.assertEqual(testDBMS.convert_integer('20h', KnobUnitTime), 72000000)
        self.assertEqual(testDBMS.convert_integer('10min', KnobUnitTime), 600000)
        self.assertEqual(testDBMS.convert_integer('1s', KnobUnitTime), 1000)


        with self.assertRaises(Exception):
            testDBMS.convert_integer('A', KnobUnitOther)

        with self.assertRaises(Exception):
            testDBMS.convert_integer('', KnobUnitOther)

    def testFormatInteger(self):
        testDBMS = PostgresParser(2)

        KnobUnitBytes = KnobUnitType()
        KnobUnitBytes.unit = 1
        KnobUnitTime = KnobUnitType()
        KnobUnitTime.unit = 2
        KnobUnitOther = KnobUnitType()
        KnobUnitOther.unit = 3

        self.assertEqual(testDBMS.format_integer(5, KnobUnitOther), 5)
        self.assertEqual(testDBMS.format_integer(0, KnobUnitOther), 0)
        self.assertEqual(testDBMS.format_integer(-1, KnobUnitOther), -1)

        self.assertEqual(testDBMS.format_integer(5120, KnobUnitBytes), '5kB')
        self.assertEqual(testDBMS.format_integer(4194304, KnobUnitBytes), '4MB')
        self.assertEqual(testDBMS.format_integer(4194500, KnobUnitBytes), '4MB')

        self.assertEqual(testDBMS.format_integer(86400000, KnobUnitTime), '1d')
        self.assertEqual(testDBMS.format_integer(72000000, KnobUnitTime), '20h')
        self.assertEqual(testDBMS.format_integer(600000, KnobUnitTime), '10min')
        self.assertEqual(testDBMS.format_integer(1000, KnobUnitTime), '1000ms')

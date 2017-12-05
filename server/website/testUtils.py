
from django.test import TestCase
from website.utils import *


class JSONUtilTest(TestCase):
    def testUtil(self):
        jsonstr = \
        """{
            "glossary": {
                "title": "example glossary",
        		"GlossDiv": {
                    "title": "S",
        			"GlossList": {
                        "GlossEntry": {
                            "ID": "SGML",
        					"SortAs": "SGML",
        					"GlossTerm": "Standard Generalized Markup Language",
        					"Acronym": "SGML",
        					"Abbrev": "ISO 8879:1986",
        					"GlossDef": {
                                "para": "A meta-markup language, used to create markup languages such as DocBook.",
        						"GlossSeeAlso": ["GML", "XML"]
                            },
        					"GlossSee": "markup"
                        }
                    }
                }
            }
        }"""

        compressedstr = \
        """{"glossary": {"title": "example glossary", "GlossDiv": {"title": "S", "GlossList": {"GlossEntry": {"ID": "SGML", "SortAs": "SGML", "GlossTerm": "Standard Generalized Markup Language", "Acronym": "SGML", "Abbrev": "ISO 8879:1986", "GlossDef": {"para": "A meta-markup language, used to create markup languages such as DocBook.", "GlossSeeAlso": ["GML", "XML"]}, "GlossSee": "markup"}}}}}"""


        results = JSONUtil.loads(jsonstr)
        self.assertEqual(results.keys()[0], "glossary")
        self.assertTrue("title" in results["glossary"].keys())
        self.assertTrue("GlossDiv" in results["glossary"].keys())
        self.assertEqual(results["glossary"]["GlossDiv"]["GlossList"]["GlossEntry"]["ID"], "SGML")
        self.assertEqual(results["glossary"]["GlossDiv"]["GlossList"]["GlossEntry"]["GlossSee"], "markup")

        resStr = JSONUtil.dumps(results)
        self.assertEqual(resStr, compressedstr)

class MediaUtilTest(TestCase):
    def testCodeGen(self):
        testSet = set()
        for i in range(1000):
            # Don't know a better way of doing this - collisions are still possible???
            code20 = MediaUtil.upload_code_generator(20)
            self.assertTrue(len(code20) == 20)
            code40 = MediaUtil.upload_code_generator(40)
            self.assertTrue(len(code40) == 40)
            digitCode = MediaUtil.upload_code_generator(40, string.digits)
            self.assertTrue(len(digitCode) == 40)
            self.assertTrue(digitCode.isdigit())

            self.assertTrue(code20 not in testSet)
            self.assertTrue(code40 not in testSet)
            self.assertTrue(digitCode not in testSet)

            testSet.add(code20)
            testSet.add(code40)
            testSet.add(digitCode)

class DataUtilTest(TestCase):
    def testAggregate(self):
        pass

    def testCombine(self):
        pass


class ConversionUtilTest(TestCase):
    def testGetRawSize(self):
        testImpl = PostgresUtilImpl(2)

        # Bytes - In Bytes
        byteTestConvert = ['1PB', '2TB', '3GB', '4MB', '5kB', '6B']
        byteAns = [1024**5, 2 * 1024**4, 3 * 1024**3, 4 * 1024**2, 5 * 1024**1, 6]
        for i in range(6):
            self.assertEqual(ConversionUtil.get_raw_size(byteTestConvert[i], system = testImpl.POSTGRES_BYTES_SYSTEM), byteAns[i])

        # Time - In Milliseconds?
        dayTestConvert = ['1000ms', '1s', '10min', '20h', '1d']
        dayAns = [1000, 1000, 600000, 72000000, 86400000]
        for i in range(5):
            self.assertEqual(ConversionUtil.get_raw_size(dayTestConvert[i], system = testImpl.POSTGRES_TIME_SYSTEM), dayAns[i])


    def testGetHumanReadable(self):
        testImpl = PostgresUtilImpl(2)

        # Bytes
        byteTestConvert = [1024**5, 2 * 1024**4, 3 * 1024**3, 4 * 1024**2, 5 * 1024**1, 6]
        byteAns = ['1PB', '2TB', '3GB', '4MB', '5kB', '6B']
        for i in range(6):
            self.assertEqual(ConversionUtil.get_human_readable(byteTestConvert[i], system = testImpl.POSTGRES_BYTES_SYSTEM), byteAns[i])

        # Time
        dayTestConvert = [1000, 1000, 600000, 72000000, 86400000]
        dayAns = ['1000ms', '1s', '10min', '20h', '1d']
        for i in range(5):
            self.assertEqual(ConversionUtil.get_raw_size(dayTestConvert[i], system = testImpl.POSTGRES_TIME_SYSTEM), dayAns[i])

class DBMSUtilImplTest(TestCase):
    def testConverts(self):
        testDBMS = PostgresUtilImpl(2)

        # Convert Bool
        self.assertEqual(testDBMS.convert_bool('on', []), BooleanType.TRUE)
        self.assertEqual(testDBMS.convert_bool('ON', []), BooleanType.TRUE)
        self.assertEqual(testDBMS.convert_bool('OFF', []), BooleanType.FALSE)
        self.assertEqual(testDBMS.convert_bool('off', []), BooleanType.FALSE)
        self.assertEqual(testDBMS.convert_bool('apavlo', []), BooleanType.FALSE)

        # Convert Enum
        EnumParam = VarType(5)
        EnumParam.enumvals = ['apple', 'oranges', 'cake']

        self.assertEqual(testDBMS.convert_enum(0, EnumParam), 'apple')
        self.assertEqual(testDBMS.convert_enum(1, EnumParam), 'oranges')
        self.assertEqual(testDBMS.convert_enum(2, EnumParam), 'cake')

        with self.assertRaises(Exception):
            testDBMS.convert_enum(3, EnumParam)



class PostgresUtilImplTest(TestCase):

    def testProperties(self):
        testDBMS = PostgresUtilImpl(2)
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

        self.assertEqual(testDBMS.configuration_filename, 'postgresql.conf')
        self.assertEqual(testDBMS.transactions_counter, 'pg_stat_database.xact_commit')

    def testParseVersionString(self):
        # TODO: Find applicable version string??
        testDBMS = PostgresUtilImpl(2)


    def testConverts(self):
        testDBMS = PostgresUtilImpl(2)

        # Convert Integer
        KnobUnitBytes = KnobUnitType()
        KnobUnitBytes.unit = 1
        KnobUnitTime = KnobUnitType()
        KnobUnitTime.unit = 2
        KnobUnitOther = KnobUnitType()
        KnobUnitOther.unit = 3


        self.assertEqual(testDBMS.convert_integer('5', K), 5)
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
        testDBMS = PostgresUtilImpl(2)

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
        self.assertEqual(testDBMS.format_integer(1000, KnobUnitTime), '1s')

class LabelUtilTest(TestCase):
    def testStyleLabels(self):
        pass

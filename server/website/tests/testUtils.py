
import string
from django.test import TestCase
from website.utils import JSONUtil, MediaUtil, DataUtil, ConversionUtil, LabelUtil
from website.parser import PostgresUtilImpl
from website.types import LabelStyleType

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
        dayTestConvert = [1000, 55000, 600000, 72000000, 86400000]
        dayAns = ['1000ms', '55000ms', '10min', '20h', '1d']
        for i in range(5):
            self.assertEqual(ConversionUtil.get_human_readable(dayTestConvert[i], system = testImpl.POSTGRES_TIME_SYSTEM), dayAns[i])



class LabelUtilTest(TestCase):
    def testStyleLabels(self):
        LabelStyleTitle = LabelStyleType()
        LabelStyleCapFirst = LabelStyleType()
        LabelStyleCapFirst.DEFAULT_STYLE = 1
        LabelStyleLower = LabelStyleType()
        LabelStyleLower.DEFAULT_STYLE = 2
        LabelStyleInvalid = LabelStyleType()
        LabelStyleInvalid.DEFAULT_STYLE = 3


        testLabelMap = { "Version": "dbms, Dbms, dBMS, Testing, testing",
        "Version2": "123",
        "Version3": "blah"}

        testTitle = LabelUtil.style_labels(testLabelMap, LabelStyleTitle.DEFAULT_STYLE)
        testCapFirst = LabelUtil.style_labels(testLabelMap, LabelStyleCapFirst.DEFAULT_STYLE)
        testLower = LabelUtil.style_labels(testLabelMap, LabelStyleLower.DEFAULT_STYLE)

        self.assertEqual(testTitle["Version"], "DBMS, DBMS, DBMS, Testing, Testing")
        self.assertEqual(testTitle["Version2"], "123")
        self.assertEqual(testTitle["Version3"], "Blah")

        self.assertEqual(testCapFirst["Version"], "DBMS, DBMS, dBMS, Testing, testing")
        self.assertEqual(testCapFirst["Version2"], "123")
        self.assertEqual(testCapFirst["Version3"], "Blah")

        self.assertEqual(testLower["Version"], "dbms, dbms, dbms, testing, testing")
        self.assertEqual(testLower["Version2"], "123")
        self.assertEqual(testLower["Version3"], "blah")

        with self.assertRaises(Exception):
            LabelUtil.style_labels(testLabelMap, LabelStyleInvalid)

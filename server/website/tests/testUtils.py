
import string
from django.test import TestCase
from website.utils import JSONUtil, MediaUtil, DataUtil, ConversionUtil, LabelUtil, TaskUtil
from website.parser.postgres import PostgresParser, Postgres96Parser
from website.types import LabelStyleType, VarType
from website.models import KnobCatalog, DBMSCatalog, MetricCatalog, Result

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
        self.assertEqual(results["glossary"]["GlossDiv"]\
        ["GlossList"]["GlossEntry"]["ID"], "SGML")
        self.assertEqual(results["glossary"]["GlossDiv"]\
        ["GlossList"]["GlossEntry"]["GlossSee"], "markup")

        resStr = JSONUtil.dumps(results)
        self.assertEqual(resStr, compressedstr)

class MediaUtilTest(TestCase):
    def testCodeGen(self):
        for i in range(1000):
            code20 = MediaUtil.upload_code_generator(20)
            self.assertEqual(len(code20), 20)
            self.assertTrue(code20.isalnum())
            code40 = MediaUtil.upload_code_generator(40)
            self.assertEqual(len(code40), 40)
            self.assertTrue(code40.isalnum())
            digitCode = MediaUtil.upload_code_generator(40, string.digits)
            self.assertEqual(len(digitCode), 40)
            self.assertTrue(digitCode.isdigit())
            letterCode = MediaUtil.upload_code_generator(60,
                            string.ascii_uppercase)
            self.assertEqual(len(letterCode), 60)
            self.assertTrue(letterCode.isalpha())

class TaskUtilTest(TestCase):
    def testGetTaskStatus(self):
        # FIXME: Actually setup celery tasks instead of a dummy class?
        testTasks = []

        (status, numComplete) = TaskUtil.get_task_status(testTasks)
        self.assertTrue(status == None and numComplete == 0)

        testTasks2 = [VarType() for i in range(5)]
        for i in range(len(testTasks2)):
            testTasks2[i].status = "SUCCESS"

        (status, numComplete) = TaskUtil.get_task_status(testTasks2)
        self.assertTrue(status == "SUCCESS" and numComplete == 5)

        testTasks3 = testTasks2
        testTasks3[3].status = "FAILURE"

        (status, numComplete) = TaskUtil.get_task_status(testTasks3)
        self.assertTrue(status == "FAILURE" and numComplete == 3)

        testTasks4 = testTasks3
        testTasks4[2].status = "REVOKED"

        (status, numComplete) = TaskUtil.get_task_status(testTasks4)
        self.assertTrue(status == "REVOKED" and numComplete == 2)

        testTasks5 = testTasks4
        testTasks5[1].status = "RETRY"

        (status, numComplete) = TaskUtil.get_task_status(testTasks5)
        self.assertTrue(status == "RETRY" and numComplete == 1)

        testTasks6 = [VarType() for i in range(10)]
        for i in range(len(testTasks6)):
            testTasks6[i].status = "PENDING" if i % 2 == 0 else "SUCCESS"

        (status, numComplete) = TaskUtil.get_task_status(testTasks6)
        self.assertTrue(status == "PENDING" and numComplete == 5)

        testTasks7 = testTasks6
        testTasks7[9].status = "STARTED"

        (status, numComplete) = TaskUtil.get_task_status(testTasks7)
        self.assertTrue(status == "STARTED" and numComplete == 4)

        testTasks8 = testTasks7
        testTasks8[9].status = "RECEIVED"

        (status, numComplete) = TaskUtil.get_task_status(testTasks8)
        self.assertTrue(status == "RECEIVED" and numComplete == 4)

        with self.assertRaises(Exception):
            testTasks9 = [VarType() for i in range(1)]
            testTasks9[0].status = "attemped"
            TaskUtil.get_task_status(testTasks9)

class DataUtilTest(TestCase):
    def testAggregate(self):
        pass

    def testCombine(self):
        import numpy as np
        testNoDupsRowLabels = np.array(["Workload-0", "Workload-1"])
        testNoDupsX = np.matrix([[0.22, 5, "string", "11:11", "fsync", True],
                                 [0.21, 6, "string", "11:12", "fsync", True]])
        testNoDupsY = np.matrix([[30, 30, 40],
                                 [10, 10, 40]])

        testX, testY, rowlabels = DataUtil.combine_duplicate_rows(testNoDupsX, testNoDupsY, testNoDupsRowLabels)

        self.assertEqual(len(testX), len(testY))
        self.assertEqual(len(testX), len(rowlabels))

        self.assertEqual(rowlabels[0], tuple([testNoDupsRowLabels[0]]))
        self.assertEqual(rowlabels[1], tuple([testNoDupsRowLabels[1]]))
        self.assertTrue((testX[0] == testNoDupsX[0]).all())
        self.assertTrue((testX[1] == testNoDupsX[1]).all())
        self.assertTrue((testY[0] == testNoDupsY[0]).all())
        self.assertTrue((testY[1] == testNoDupsY[1]).all())

        testRowLabels = np.array(["Workload-0",
                                  "Workload-1",
                                  "Workload-2",
                                  "Workload-3"])
        testXMat = np.matrix([[0.22, 5, "string", "timestamp", "enum", True],
                              [0.3, 5, "rstring", "timestamp2", "enum", False],
                              [0.22, 5, "string", "timestamp", "enum", True],
                              [0.3, 5, "r", "timestamp2", "enum", False]])
        testYMat = np.matrix([[20, 30, 40],
                              [30, 30, 40],
                              [20, 30, 40],
                              [32, 30, 40]])

        testX, testY, rowlabels = DataUtil.combine_duplicate_rows(testXMat, testYMat, testRowLabels)

        self.assertTrue(len(testX) <= len(testXMat))
        self.assertTrue(len(testY) <= len(testYMat))
        self.assertEqual(len(testX), len(testY))
        self.assertEqual(len(testX), len(rowlabels))

        rowlabelsSet = set(rowlabels)
        self.assertTrue(tuple(["Workload-0", "Workload-2"]) in rowlabelsSet)
        self.assertTrue(("Workload-1",) in rowlabelsSet)
        self.assertTrue(("Workload-3",) in rowlabelsSet)

        rows = set()
        for i in testX:
            self.assertTrue(tuple(i) not in rows)
            self.assertTrue(i in testXMat)
            rows.add(tuple(i))

        rowys = set()
        for i in testY:
            self.assertTrue(tuple(i) not in rowys)
            self.assertTrue(i in testYMat)
            rowys.add(tuple(i))


class ConversionUtilTest(TestCase):
    def testGetRawSize(self):
        testImpl = PostgresParser(2)

        # Bytes - In Bytes
        byteTestConvert = ['1PB', '2TB', '3GB', '4MB', '5kB', '6B']
        byteAns = [1024**5, 2 * 1024**4, 3 * 1024**3, 4 * 1024**2, 5 * 1024**1, 6]
        for i in range(6):
            self.assertEqual(ConversionUtil.get_raw_size(
                                byteTestConvert[i],
                                system = testImpl.POSTGRES_BYTES_SYSTEM),
                                byteAns[i])

        # Time - In Milliseconds?
        dayTestConvert = ['1000ms', '1s', '10min', '20h', '1d']
        dayAns = [1000, 1000, 600000, 72000000, 86400000]
        for i in range(5):
            self.assertEqual(ConversionUtil.get_raw_size(
                                dayTestConvert[i],
                                system = testImpl.POSTGRES_TIME_SYSTEM),
                                dayAns[i])


    def testGetHumanReadable(self):
        testImpl = PostgresParser(2)

        # Bytes
        byteTestConvert = [1024**5, 2 * 1024**4, 3 * 1024**3,
                           4 * 1024**2, 5 * 1024**1, 6]
        byteAns = ['1PB', '2TB', '3GB', '4MB', '5kB', '6B']
        for i in range(6):
            self.assertEqual(ConversionUtil.get_human_readable(
                                byteTestConvert[i],
                                system = testImpl.POSTGRES_BYTES_SYSTEM),
                                byteAns[i])

        # Time
        dayTestConvert = [500, 1000, 55000, 600000, 72000000, 86400000]
        dayAns = ['500ms', '1s', '55s', '10min', '20h', '1d']
        for i in range(5):
            self.assertEqual(dayAns[i],
            ConversionUtil.get_human_readable(dayTestConvert[i],
                            system = testImpl.POSTGRES_TIME_SYSTEM))

class LabelUtilTest(TestCase):
    def testStyleLabels(self):
        LabelStyle = LabelStyleType()

        testLabelMap = {"Name": "Postgres",
                        "Test": "LabelUtils",
                        "DBMS": "dbms",
                        "??"  : "Dbms",
                        "???" : "DBms",
                        "CapF": "random Word"}

        resTitleLabelMap = LabelUtil.style_labels(testLabelMap,
                                style = LabelStyle.TITLE)

        testKeys = ["Name", "Test", "DBMS", "??", "???", "CapF"]
        titleAns = ["Postgres", "Labelutils", "DBMS", "DBMS", "DBMS",
                    "Random Word"]

        for i, key in enumerate(testKeys):
            self.assertEqual(resTitleLabelMap[key], titleAns[i])

        resCapfirstLabelMap = LabelUtil.style_labels(testLabelMap,
                                style = LabelStyle.CAPFIRST)

        capAns = ["Postgres", "LabelUtils", "DBMS", "DBMS", "DBMS",
                  "Random Word"]

        for i, key in enumerate(testKeys):
            if (key == "???"): # DBms -> DBMS or DBms?
                continue
            self.assertEqual(resCapfirstLabelMap[key], capAns[i])

        resLowerLabelMap = LabelUtil.style_labels(testLabelMap,
                                style = LabelStyle.LOWER)

        lowerAns = ["postgres", "labelutils", "dbms", "dbms", "dbms",
                    "random word"]

        for i, key in enumerate(testKeys):
            self.assertEqual(resLowerLabelMap[key], lowerAns[i])

        with self.assertRaises(Exception):
            resExcept = LabelUtil.style_labels(testLabelMap,
                                               style = LabelStyle.Invalid)

#
# OtterTune - test_utils.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

import string
import numpy as np
from django.test import TestCase
from website.utils import JSONUtil, MediaUtil, DataUtil, ConversionUtil, LabelUtil, TaskUtil
from website.parser.postgres import PostgresParser
from website.types import LabelStyleType, VarType
from website.models import Result, DBMSCatalog


class JSONUtilTest(TestCase):
    def test_util(self):
        json_str = \
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
                                "para": "A meta-markup language",
                                "GlossSeeAlso": ["GML", "XML"]
                            },
                            "GlossSee": "markup"
                        }
                    }
                }
            }
        }"""

        compress_str = """{"glossary": {"title": "example glossary",
         "GlossDiv": {"title": "S", "GlossList": {"GlossEntry": {"ID": "SGML",
          "SortAs": "SGML", "GlossTerm": "Standard Generalized Markup
           Language", "Acronym": "SGML", "Abbrev": "ISO 8879:1986", "GlossDef":
            {"para": "A meta-markup language", "GlossSeeAlso": ["GML", "XML"]}, "GlossSee":
              "markup"}}}}}"""

        results = JSONUtil.loads(json_str)
        self.assertEqual(list(results.keys())[0], "glossary")
        self.assertTrue("title" in list(results["glossary"].keys()))
        self.assertTrue("GlossDiv" in list(results["glossary"].keys()))
        self.assertEqual(results["glossary"]["GlossDiv"]
                         ["GlossList"]["GlossEntry"]["ID"], "SGML")
        self.assertEqual(results["glossary"]["GlossDiv"]
                         ["GlossList"]["GlossEntry"]["GlossSee"], "markup")

        result_str = "".join(JSONUtil.dumps(results).split())
        self.assertEqual(result_str, "".join(compress_str.split()))


class MediaUtilTest(TestCase):
    def test_codegen(self):
        code20 = MediaUtil.upload_code_generator(20)
        self.assertEqual(len(code20), 20)
        self.assertTrue(code20.isalnum())
        code40 = MediaUtil.upload_code_generator(40)
        self.assertEqual(len(code40), 40)
        self.assertTrue(code40.isalnum())
        digit_code = MediaUtil.upload_code_generator(40, string.digits)
        self.assertEqual(len(digit_code), 40)
        self.assertTrue(digit_code.isdigit())
        letter_code = MediaUtil.upload_code_generator(60,
                                                      string.ascii_uppercase)
        self.assertEqual(len(letter_code), 60)
        self.assertTrue(letter_code.isalpha())


class TaskUtilTest(TestCase):
    def test_get_task_status(self):
        # FIXME: Actually setup celery tasks instead of a dummy class?
        test_tasks = []

        (status, num_complete) = TaskUtil.get_task_status(test_tasks)
        self.assertTrue(status is None and num_complete == 0)

        test_tasks2 = [VarType() for i in range(5)]
        for task in test_tasks2:
            task.status = "SUCCESS"

        (status, num_complete) = TaskUtil.get_task_status(test_tasks2)
        self.assertTrue(status == "SUCCESS" and num_complete == 5)

        test_tasks3 = test_tasks2
        test_tasks3[3].status = "FAILURE"

        (status, num_complete) = TaskUtil.get_task_status(test_tasks3)
        self.assertTrue(status == "FAILURE" and num_complete == 3)

        test_tasks4 = test_tasks3
        test_tasks4[2].status = "REVOKED"

        (status, num_complete) = TaskUtil.get_task_status(test_tasks4)
        self.assertTrue(status == "REVOKED" and num_complete == 2)

        test_tasks5 = test_tasks4
        test_tasks5[1].status = "RETRY"

        (status, num_complete) = TaskUtil.get_task_status(test_tasks5)
        self.assertTrue(status == "RETRY" and num_complete == 1)

        test_tasks6 = [VarType() for i in range(10)]
        for i, task in enumerate(test_tasks6):
            task.status = "PENDING" if i % 2 == 0 else "SUCCESS"

        (status, num_complete) = TaskUtil.get_task_status(test_tasks6)
        self.assertTrue(status == "PENDING" and num_complete == 5)

        test_tasks7 = test_tasks6
        test_tasks7[9].status = "STARTED"

        (status, num_complete) = TaskUtil.get_task_status(test_tasks7)
        self.assertTrue(status == "STARTED" and num_complete == 4)

        test_tasks8 = test_tasks7
        test_tasks8[9].status = "RECEIVED"

        (status, num_complete) = TaskUtil.get_task_status(test_tasks8)
        self.assertTrue(status == "RECEIVED" and num_complete == 4)

        with self.assertRaises(Exception):
            test_tasks9 = [VarType() for i in range(1)]
            test_tasks9[0].status = "attemped"
            TaskUtil.get_task_status(test_tasks9)


class DataUtilTest(TestCase):

    fixtures = ['test_website.json', 'postgres-96_knobs.json']

    def test_aggregate(self):

        workload2 = Result.objects.filter(workload=2)
        num_results = Result.objects.filter(workload=2).count()
        knobs = list(JSONUtil.loads(workload2[0].knob_data.data).keys())
        metrics = list(JSONUtil.loads(workload2[0].metric_data.data).keys())
        num_knobs = len(knobs)
        num_metrics = len(metrics)

        test_result = DataUtil.aggregate_data(workload2)

        self.assertTrue('X_matrix' in list(test_result.keys()))
        self.assertTrue('y_matrix' in list(test_result.keys()))
        self.assertTrue('rowlabels' in list(test_result.keys()))
        self.assertTrue('X_columnlabels' in list(test_result.keys()))
        self.assertTrue('y_columnlabels' in list(test_result.keys()))

        self.assertEqual(test_result['X_columnlabels'], knobs)
        self.assertEqual(test_result['y_columnlabels'], metrics)
        self.assertEqual(test_result['X_matrix'].shape[0], num_results)
        self.assertEqual(test_result['y_matrix'].shape[0], num_results)
        self.assertEqual(test_result['X_matrix'].shape[1], num_knobs)
        self.assertEqual(test_result['y_matrix'].shape[1], num_metrics)

    def test_combine(self):
        test_dedup_row_labels = np.array(["Workload-0", "Workload-1"])
        test_dedup_x = np.matrix([[0.22, 5, "string", "11:11", "fsync", True],
                                  [0.21, 6, "string", "11:12", "fsync", True]])
        test_dedup_y = np.matrix([[30, 30, 40],
                                  [10, 10, 40]])

        test_x, test_y, row_labels = DataUtil.combine_duplicate_rows(
            test_dedup_x, test_dedup_y, test_dedup_row_labels)

        self.assertEqual(len(test_x), len(test_y))
        self.assertEqual(len(test_x), len(row_labels))

        self.assertEqual(row_labels[0], tuple([test_dedup_row_labels[0]]))
        self.assertEqual(row_labels[1], tuple([test_dedup_row_labels[1]]))
        self.assertTrue((test_x[0] == test_dedup_x[0]).all())
        self.assertTrue((test_x[1] == test_dedup_x[1]).all())
        self.assertTrue((test_y[0] == test_dedup_y[0]).all())
        self.assertTrue((test_y[1] == test_dedup_y[1]).all())

        test_row_labels = np.array(["Workload-0",
                                    "Workload-1",
                                    "Workload-2",
                                    "Workload-3"])
        test_x_matrix = np.matrix([[0.22, 5, "string", "timestamp", "enum", True],
                                   [0.3, 5, "rstring", "timestamp2", "enum", False],
                                   [0.22, 5, "string", "timestamp", "enum", True],
                                   [0.3, 5, "r", "timestamp2", "enum", False]])
        test_y_matrix = np.matrix([[20, 30, 40],
                                   [30, 30, 40],
                                   [20, 30, 40],
                                   [32, 30, 40]])

        test_x, test_y, row_labels = DataUtil.combine_duplicate_rows(
            test_x_matrix, test_y_matrix, test_row_labels)

        self.assertTrue(len(test_x) <= len(test_x_matrix))
        self.assertTrue(len(test_y) <= len(test_y_matrix))
        self.assertEqual(len(test_x), len(test_y))
        self.assertEqual(len(test_x), len(row_labels))

        row_labels_set = set(row_labels)
        self.assertTrue(tuple(["Workload-0", "Workload-2"]) in row_labels_set)
        self.assertTrue(("Workload-1",) in row_labels_set)
        self.assertTrue(("Workload-3",) in row_labels_set)

        rows = set()
        for i in test_x:
            self.assertTrue(tuple(i) not in rows)
            self.assertTrue(i in test_x_matrix)
            rows.add(tuple(i))

        rowys = set()
        for i in test_y:
            self.assertTrue(tuple(i) not in rowys)
            self.assertTrue(i in test_y_matrix)
            rowys.add(tuple(i))

    def test_no_featured_categorical(self):
        featured_knobs = ['global.backend_flush_after',
                          'global.bgwriter_delay',
                          'global.wal_writer_delay',
                          'global.work_mem']
        postgres96 = DBMSCatalog.objects.get(pk=1)
        categorical_info = DataUtil.dummy_encoder_helper(featured_knobs,
                                                         dbms=postgres96)
        self.assertEqual(len(categorical_info['n_values']), 0)
        self.assertEqual(len(categorical_info['categorical_features']), 0)
        self.assertEqual(categorical_info['cat_columnlabels'], [])
        self.assertEqual(categorical_info['noncat_columnlabels'], featured_knobs)

    def test_featured_categorical(self):
        featured_knobs = ['global.backend_flush_after',
                          'global.bgwriter_delay',
                          'global.wal_writer_delay',
                          'global.work_mem',
                          'global.wal_sync_method']  # last knob categorical
        postgres96 = DBMSCatalog.objects.get(pk=1)
        categorical_info = DataUtil.dummy_encoder_helper(featured_knobs,
                                                         dbms=postgres96)
        self.assertEqual(len(categorical_info['n_values']), 1)
        self.assertEqual(categorical_info['n_values'][0], 4)
        self.assertEqual(len(categorical_info['categorical_features']), 1)
        self.assertEqual(categorical_info['categorical_features'][0], 4)
        self.assertEqual(categorical_info['cat_columnlabels'], ['global.wal_sync_method'])
        self.assertEqual(categorical_info['noncat_columnlabels'], featured_knobs[:-1])


class ConversionUtilTest(TestCase):
    def test_get_raw_size(self):
        # Bytes - In Bytes
        byte_test_convert = ['1PB', '2TB', '3GB', '4MB', '5kB', '6B']
        byte_ans = [1024**5, 2 * 1024**4, 3 * 1024**3, 4 * 1024**2, 5 * 1024**1, 6]
        for i, byte_test in enumerate(byte_test_convert):
            byte_conversion = ConversionUtil.get_raw_size(
                byte_test, system=PostgresParser.POSTGRES_BYTES_SYSTEM)
            self.assertEqual(byte_conversion, byte_ans[i])

        # Time - In Milliseconds
        day_test_convert = ['1000ms', '1s', '10min', '20h', '1d']
        day_ans = [1000, 1000, 600000, 72000000, 86400000]
        for i, day_test in enumerate(day_test_convert):
            day_conversion = ConversionUtil.get_raw_size(
                day_test, system=PostgresParser.POSTGRES_TIME_SYSTEM)
            self.assertEqual(day_conversion, day_ans[i])

    def test_get_human_readable(self):
        # Bytes
        byte_test_convert = [1024**5, 2 * 1024**4, 3 * 1024**3,
                             4 * 1024**2, 5 * 1024**1, 6]
        byte_ans = ['1PB', '2TB', '3GB', '4MB', '5kB', '6B']
        for i, byte_test in enumerate(byte_test_convert):
            byte_readable = ConversionUtil.get_human_readable(
                byte_test, system=PostgresParser.POSTGRES_BYTES_SYSTEM)
            self.assertEqual(byte_readable, byte_ans[i])

        # Time
        day_test_convert = [500, 1000, 55000, 600000, 72000000, 86400000]
        day_ans = ['500ms', '1s', '55s', '10min', '20h', '1d']
        for i, day_test in enumerate(day_test_convert):
            day_readable = ConversionUtil.get_human_readable(
                day_test, system=PostgresParser.POSTGRES_TIME_SYSTEM)
            self.assertEqual(day_readable, day_ans[i])


class LabelUtilTest(TestCase):
    def test_style_labels(self):
        label_style = LabelStyleType()

        test_label_map = {"Name": "Postgres",
                          "Test": "LabelUtils",
                          "DBMS": "dbms",
                          "??": "Dbms",
                          "???": "DBms",
                          "CapF": "random Word"}

        res_title_label_map = LabelUtil.style_labels(test_label_map,
                                                     style=label_style.TITLE)

        test_keys = ["Name", "Test", "DBMS", "??", "???", "CapF"]
        title_ans = ["Postgres", "Labelutils", "DBMS", "DBMS", "DBMS",
                     "Random Word"]

        for i, key in enumerate(test_keys):
            self.assertEqual(res_title_label_map[key], title_ans[i])

        res_capfirst_label_map = LabelUtil.style_labels(test_label_map,
                                                        style=label_style.CAPFIRST)

        cap_ans = ["Postgres", "LabelUtils", "DBMS", "DBMS", "DBMS",
                   "Random Word"]

        for i, key in enumerate(test_keys):
            if (key == "???"):  # DBms -> DBMS or DBms?
                continue
            self.assertEqual(res_capfirst_label_map[key], cap_ans[i])

        res_lower_label_map = LabelUtil.style_labels(test_label_map,
                                                     style=label_style.LOWER)

        lower_ans = ["postgres", "labelutils", "dbms", "dbms", "dbms",
                     "random word"]

        for i, key in enumerate(test_keys):
            self.assertEqual(res_lower_label_map[key], lower_ans[i])

        with self.assertRaises(Exception):
            LabelUtil.style_labels(test_label_map,
                                   style=label_style.Invalid)

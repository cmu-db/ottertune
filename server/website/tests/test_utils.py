#
# OtterTune - test_utils.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

import string
from collections import OrderedDict

import numpy as np
from django.test import TestCase

from website.utils import JSONUtil, MediaUtil, DataUtil, ConversionUtil, LabelUtil, TaskUtil
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
        postgresdb = DBMSCatalog.objects.get(pk=1)
        categorical_info = DataUtil.dummy_encoder_helper(featured_knobs,
                                                         dbms=postgresdb)
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
        postgresdb = DBMSCatalog.objects.get(pk=1)
        categorical_info = DataUtil.dummy_encoder_helper(featured_knobs,
                                                         dbms=postgresdb)
        self.assertEqual(len(categorical_info['n_values']), 1)
        self.assertEqual(categorical_info['n_values'][0], 4)
        self.assertEqual(len(categorical_info['categorical_features']), 1)
        self.assertEqual(categorical_info['categorical_features'][0], 4)
        self.assertEqual(categorical_info['cat_columnlabels'], ['global.wal_sync_method'])
        self.assertEqual(categorical_info['noncat_columnlabels'], featured_knobs[:-1])


class ConversionUtilTest(TestCase):

    def setUp(self):
        self.bytes_map = OrderedDict(
            [(suffix, factor) for factor, suffix in ConversionUtil.DEFAULT_BYTES_SYSTEM])
        self.ms_map = OrderedDict(
            [(suffix, factor) for factor, suffix in ConversionUtil.DEFAULT_TIME_SYSTEM])

        self.from_hr_bytes_simple = ['1PB', '2TB', '3GB', '4MB', '1024MB', '5kB', '6B']
        self.as_bytes_simple = [1024**5, 2 * 1024**4, 3 * 1024**3, 4 * 1024**2, 1024**3,
                                5 * 1024, 6]
        self.bytes_to_hr_simple = ['1PB', '2TB', '3GB', '4MB', '1GB', '5kB', '6B']
        self.assertListEqual(
            [len(l) for l in (self.from_hr_bytes_simple, self.as_bytes_simple,
                              self.bytes_to_hr_simple)], [len(self.from_hr_bytes_simple)] * 3)

        self.from_hr_times_simple = ['500ms', '1000ms', '1s', '55s', '10min', '20h', '1d']
        self.as_ms_simple = [500, 1000, 1000, 55000, 600000, 72000000, 86400000]
        self.ms_to_hr_simple = ['500ms', '1s', '1s', '55s', '10min', '20h', '1d']
        self.assertListEqual(
            [len(l) for l in (self.from_hr_times_simple, self.as_ms_simple,
                              self.ms_to_hr_simple)], [len(self.from_hr_times_simple)] * 3)

        extra_bytes = [3 * factor for factor in self.bytes_map.values()]
        neb = len(extra_bytes)

        self.test_bytes_lengths = []

        self.from_hr_bytes = [
            '1PB', '43PB', '1023PB', '1024PB', '1025PB',
            '1TB', '43TB', '1023TB', '1024TB', '1025TB',
            '1GB', '43GB', '1023GB', '1024GB', '1025GB',
            '1MB', '43MB', '1023MB', '1024MB', '1025MB',
            '1kB', '43kB', '1023kB', '1024kB', '1025kB',
            '1B', '43B', '1023B', '1024B', '1025B',
            '46170898432MB', '45088768MB', '44032MB',
            '44032kB', '44032B', '43kB',
        ] + ['43{}'.format(suffix) for suffix in list(self.bytes_map.keys())[1:]] + \
            ['{}B'.format(sum(extra_bytes[i:])) for i in range(neb)]
        self.test_bytes_lengths.append(len(self.from_hr_bytes))

        self.as_bytes = [
            1024**5, 43 * 1024**5, 1023 * 1024**5, 1024 * 1024**5, 1025 * 1024**5,
            1024**4, 43 * 1024**4, 1023 * 1024**4, 1024 * 1024**4, 1025 * 1024**4,
            1024**3, 43 * 1024**3, 1023 * 1024**3, 1024 * 1024**3, 1025 * 1024**3,
            1024**2, 43 * 1024**2, 1023 * 1024**2, 1024 * 1024**2, 1025 * 1024**2,
            1024**1, 43 * 1024**1, 1023 * 1024**1, 1024 * 1024**1, 1025 * 1024**1,
            1024**0, 43 * 1024**0, 1023 * 1024**0, 1024 * 1024**0, 1025 * 1024**0,
            46170898432 * 1024**2, 45088768 * 1024**2, 44032 * 1024**2,
            44032 * 1024, 44032, 43 * 1024,
        ] + [43 * 1024**i for i in range(len(self.bytes_map) - 1)[::-1]] + \
            [sum(extra_bytes[i:]) for i in range(neb)]
        self.test_bytes_lengths.append(len(self.as_bytes))

        self.bytes_to_hr = [
            '1PB', '43PB', '1023PB', '1024PB', '1025PB',
            '1TB', '43TB', '1023TB', '1PB', '1PB',
            '1GB', '43GB', '1023GB', '1TB', '1TB',
            '1MB', '43MB', '1023MB', '1GB', '1GB',
            '1kB', '43kB', '1023kB', '1MB', '1MB',
            '1B', '43B', '1023B', '1kB', '1kB',
            '43PB', '43TB', '43GB',
            '43MB', '43kB', '43kB',
        ] + ['43{}'.format(suffix) for suffix in list(self.bytes_map.keys())[1:]] + \
            ['3{}'.format(suffix) for suffix in self.bytes_map.keys()]
        self.test_bytes_lengths.append(len(self.bytes_to_hr))

        self.bytes_to_hr2 = [
            '1PB', '43PB', '1023PB', '1024PB', '1025PB',
            '1TB', '43TB', '1023TB', '1PB', '1025TB',
            '1GB', '43GB', '1023GB', '1TB', '1025GB',
            '1MB', '43MB', '1023MB', '1GB', '1025MB',
            '1kB', '43kB', '1023kB', '1MB', '1025kB',
            '1B', '43B', '1023B', '1kB', '1025B',
            '43PB', '43TB', '43GB',
            '43MB', '43kB', '43kB',
        ] + ['43{}'.format(suffix) for suffix in list(self.bytes_map.keys())[1:]] + \
            ['{}kB'.format(sum(extra_bytes[i:]) // 1024) for i in range(neb - 1)] + \
            ['{}B'.format(extra_bytes[-1])]
        self.test_bytes_lengths.append(len(self.bytes_to_hr2))

        self.min_bytes_suffixes = (25 * ['kB']) + (11 * ['B']) + \
            list(self.bytes_map.keys())[:-1] + \
            ((neb - 1) * ['kB']) + ['B']
        self.test_bytes_lengths.append(len(self.min_bytes_suffixes))

        self.assertListEqual(self.test_bytes_lengths,
                             [self.test_bytes_lengths[0]] * len(self.test_bytes_lengths))

        self.test_ms_lengths = []

        extra_ms = [3 * factor for factor in self.ms_map.values()]
        nem = len(extra_ms)

        self.from_hr_times = [
            '1d', '5d', '6d', '7d', '8d',
            '1h', '5h', '23h', '24h', '25h',
            '1min', '5min', '59min', '60min', '61min',
            '1s', '5s', '59s', '60s', '61s',
            '1ms', '5ms', '999ms', '1000ms', '1001ms',
            '7200min', '300min', '300s', '5000ms', '5s',
        ] + ['5{}'.format(suffix) for suffix in list(self.ms_map.keys())[1:]] + \
            ['{}ms'.format(sum(extra_ms[i:])) for i in range(nem)]
        self.test_ms_lengths.append(len(self.from_hr_times))

        self.as_ms = [v * 86400000 for v in (1, 5, 6, 7, 8)] + \
            [v * 3600000 for v in (1, 5, 23, 24, 25)] + \
            [v * 60000 for v in (1, 5, 59, 60, 61)] + \
            [v * 1000 for v in (1, 5, 59, 60, 61)] + \
            [v * 1 for v in (1, 5, 999, 1000, 1001)] + \
            [432000000, 18000000, 300000, 5000, 5000] + \
            [5 * v for v in (3600000, 60000, 1000, 1)] + \
            [sum(extra_ms[i:]) for i in range(nem)]
        self.test_ms_lengths.append(len(self.as_ms))

        self.ms_to_hr = [
            '1d', '5d', '6d', '7d', '8d',
            '1h', '5h', '23h', '1d', '1d',
            '1min', '5min', '59min', '1h', '1h',
            '1s', '5s', '59s', '1min', '1min',
            '1ms', '5ms', '999ms', '1s', '1s',
            '5d', '5h', '5min', '5s', '5s',
        ] + ['5{}'.format(suffix) for suffix in list(self.ms_map.keys())[1:]] + \
            ['3{}'.format(suffix) for suffix in self.ms_map.keys()]
        self.test_ms_lengths.append(len(self.ms_to_hr))

        self.ms_to_hr2 = [
            '1d', '5d', '6d', '7d', '8d',
            '1h', '5h', '23h', '1d', '25h',
            '1min', '5min', '59min', '1h', '61min',
            '1s', '5s', '59s', '1min', '61s',
            '1ms', '5ms', '999ms', '1s', '1001ms',
            '5d', '5h', '5min', '5s', '5s',
        ] + ['5{}'.format(suffix) for suffix in list(self.ms_map.keys())[1:]] + \
            ['{}s'.format(sum(extra_ms[i:]) // 1000) for i in range(nem - 1)] + \
            ['{}ms'.format(extra_ms[-1])]
        self.test_ms_lengths.append(len(self.ms_to_hr2))

        self.min_time_suffixes = (20 * ['s']) + (10 * ['ms']) + list(self.ms_map.keys())[:-1] + \
            ((nem - 1) * ['s']) + ['ms']
        self.test_ms_lengths.append(len(self.min_time_suffixes))

        self.assertListEqual(self.test_ms_lengths,
                             [self.test_ms_lengths[0]] * len(self.test_ms_lengths))

    def test_default_system(self):
        expected_byte_units = ('B', 'kB', 'MB', 'GB', 'TB', 'PB')
        expected_byte_values = (1, 1024, 1024**2, 1024**3, 1024**4, 1024**5)
        self.assertEqual(set(self.bytes_map.keys()), set(expected_byte_units))
        for unit, exp_val in zip(expected_byte_units, expected_byte_values):
            self.assertEqual(self.bytes_map[unit], exp_val)

        expected_time_units = ('ms', 's', 'min', 'h', 'd')
        expected_time_values = (1, 1000, 60000, 3600000, 86400000)
        self.assertEqual(set(self.ms_map.keys()), set(expected_time_units))
        for unit, exp_val in zip(expected_time_units, expected_time_values):
            self.assertEqual(self.ms_map[unit], exp_val)

    def test_get_raw_size_simple(self):
        # Bytes
        for hr_value, exp_value in zip(self.from_hr_bytes_simple, self.as_bytes_simple):
            value = ConversionUtil.get_raw_size(
                hr_value, system=ConversionUtil.DEFAULT_BYTES_SYSTEM)
            self.assertEqual(value, exp_value)

        # Time
        for hr_value, exp_value in zip(self.from_hr_times_simple, self.as_ms_simple):
            value = ConversionUtil.get_raw_size(
                hr_value, system=ConversionUtil.DEFAULT_TIME_SYSTEM)
            self.assertEqual(value, exp_value)

    def test_get_raw_size(self):
        # Bytes
        for hr_value, exp_value in zip(self.from_hr_bytes, self.as_bytes):
            byte_conversion = ConversionUtil.get_raw_size(
                hr_value, system=ConversionUtil.DEFAULT_BYTES_SYSTEM)
            self.assertEqual(byte_conversion, exp_value)

        # Time
        for hr_value, exp_value in zip(self.from_hr_times, self.as_ms):
            time_conversion = ConversionUtil.get_raw_size(
                hr_value, system=ConversionUtil.DEFAULT_TIME_SYSTEM)
            self.assertEqual(time_conversion, exp_value)

    def test_get_human_readable_simple(self):
        # Bytes
        for raw_value, exp_value in zip(self.as_bytes_simple, self.bytes_to_hr_simple):
            value = ConversionUtil.get_human_readable(
                raw_value, system=ConversionUtil.DEFAULT_BYTES_SYSTEM)
            self.assertEqual(value, exp_value)
            value2 = ConversionUtil.get_human_readable2(
                raw_value, system=ConversionUtil.DEFAULT_BYTES_SYSTEM,
                min_suffix='B')
            self.assertEqual(value2, exp_value)

        value = ConversionUtil.get_human_readable2(
            44, system=ConversionUtil.DEFAULT_BYTES_SYSTEM,
            min_suffix='kB')
        self.assertEqual(value, '44B')

        # Time
        for raw_value, exp_value in zip(self.as_ms_simple, self.ms_to_hr_simple):
            value = ConversionUtil.get_human_readable(
                raw_value, system=ConversionUtil.DEFAULT_TIME_SYSTEM)
            self.assertEqual(value, exp_value)
            value2 = ConversionUtil.get_human_readable2(
                raw_value, system=ConversionUtil.DEFAULT_TIME_SYSTEM,
                min_suffix='ms')
            self.assertEqual(value2, exp_value)

        value = ConversionUtil.get_human_readable2(
            44, system=ConversionUtil.DEFAULT_TIME_SYSTEM,
            min_suffix='s')
        self.assertEqual(value, '44ms')

    def test_get_human_readable(self):
        # Bytes
        for i, raw_bytes in enumerate(self.as_bytes):
            exp_hr = self.bytes_to_hr[i]
            exp_hr2 = self.bytes_to_hr2[i]
            min_suffix = self.min_bytes_suffixes[i]
            hr_value = ConversionUtil.get_human_readable(
                raw_bytes, system=ConversionUtil.DEFAULT_BYTES_SYSTEM)
            hr_value2 = ConversionUtil.get_human_readable2(
                raw_bytes, system=ConversionUtil.DEFAULT_BYTES_SYSTEM,
                min_suffix=min_suffix)
            self.assertEqual(hr_value, exp_hr)
            self.assertEqual(hr_value2, exp_hr2)

        # Time
        for i, raw_time in enumerate(self.as_ms):
            exp_hr = self.ms_to_hr[i]
            exp_hr2 = self.ms_to_hr2[i]
            min_suffix = self.min_time_suffixes[i]
            hr_value = ConversionUtil.get_human_readable(
                raw_time, system=ConversionUtil.DEFAULT_TIME_SYSTEM)
            hr_value2 = ConversionUtil.get_human_readable2(
                raw_time, system=ConversionUtil.DEFAULT_TIME_SYSTEM,
                min_suffix=min_suffix)
            self.assertEqual(hr_value, exp_hr)
            self.assertEqual(hr_value2, exp_hr2)


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
            if key == "???":  # DBms -> DBMS or DBms?
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

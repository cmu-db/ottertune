#
# OtterTune - test_parser.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from abc import ABCMeta, abstractmethod
import mock
from django.test import TestCase
from website.db import parser, target_objectives
from website.types import BooleanType, DBMSType, VarType, KnobUnitType, MetricType
from website.models import DBMSCatalog, KnobCatalog, MetricCatalog


class BaseParserTests(object, metaclass=ABCMeta):

    def setUp(self):
        self.test_dbms = None

    def test_convert_bool(self):
        mock_bool_knob = mock.Mock(spec=KnobCatalog)

        for bool_val in self.test_dbms.valid_true_val:
            self.assertEqual(BooleanType.TRUE,
                             self.test_dbms.convert_bool(bool_val, mock_bool_knob))

        for bool_val in self.test_dbms.valid_false_val:
            self.assertEqual(BooleanType.FALSE,
                             self.test_dbms.convert_bool(bool_val, mock_bool_knob))

        with self.assertRaises(Exception):
            self.test_dbms.convert_bool('ThisShouldNeverBeABool', mock_bool_knob)

    def test_convert_enum(self):
        mock_enum_knob = mock.Mock(spec=KnobCatalog)
        mock_enum_knob.vartype = VarType.ENUM
        mock_enum_knob.enumvals = 'apples,oranges,cake'
        mock_enum_knob.name = 'Test'

        self.assertEqual(self.test_dbms.convert_enum('apples', mock_enum_knob), 0)
        self.assertEqual(self.test_dbms.convert_enum('oranges', mock_enum_knob), 1)
        self.assertEqual(self.test_dbms.convert_enum('cake', mock_enum_knob), 2)

        with self.assertRaises(Exception):
            self.test_dbms.convert_enum('jackyl', mock_enum_knob)

    def test_convert_integer(self):
        mock_int_knob = mock.Mock(spec=KnobCatalog)
        mock_int_knob.vartype = VarType.INTEGER
        mock_int_knob.name = 'Test'

        test_int = ['42', '-1', '0', '1', '42.0', '42.5', '42.7']
        test_int_ans = [42, -1, 0, 1, 42, 42, 42]

        for test_int, test_int_ans in zip(test_int, test_int_ans):
            test_int_actual = self.test_dbms.convert_integer(test_int, mock_int_knob)
            self.assertEqual(test_int_actual, test_int_ans)

        with self.assertRaises(Exception):
            self.test_dbms.convert_integer('notInt', mock_int_knob)

    def test_convert_real(self):
        mock_real_knob = mock.Mock(spec=KnobCatalog)
        mock_real_knob.vartype = VarType.REAL
        mock_real_knob.name = 'Test'

        test_real = ['42.0', '42.2', '42.5', '42.7', '-1', '0', '1']
        test_real_ans = [42.0, 42.2, 42.5, 42.7, -1.0, 0.0, 1.0]

        for test_real, test_real_ans in zip(test_real, test_real_ans):
            test_real_actual = self.test_dbms.convert_real(test_real, mock_real_knob)
            self.assertEqual(test_real_actual, test_real_ans)

        with self.assertRaises(Exception):
            self.test_dbms.convert_real('notReal', mock_real_knob)

    def test_convert_string(self):
        # NOTE: Hasn't been used in any currently supported database
        pass

    def test_convert_timestamp(self):
        # NOTE: Hasn't been used in any currently supported database
        pass

    @abstractmethod
    def test_convert_dbms_knobs(self):
        pass

    @abstractmethod
    def test_convert_dbms_metrics(self):
        pass

    @abstractmethod
    def test_extract_valid_variables(self):
        pass

    def test_parse_helper(self):
        test_view_vars = {'local': {'FAKE_KNOB': 'FAKE'}}
        test_scope = 'global'
        valid_vars = {}
        test_parse = self.test_dbms.parse_helper(test_scope, valid_vars, test_view_vars)

        self.assertEqual(len(list(test_parse.keys())), 1)
        self.assertEqual(test_parse.get('local.FAKE_KNOB'), ['FAKE'])

    def test_parse_dbms_variables(self):
        test_dbms_vars = {'global': {'GlobalView1':
                                     {'cpu_tuple_cost': 0.01,
                                      'random_page_cost': 0.22},
                                     'GlobalView2':
                                     {'cpu_tuple_cost': 0.05,
                                      'random_page_cost': 0.25}},
                          'local': {'CustomerTable':
                                    {'LocalView1':
                                     {'LocalObj1':
                                      {'cpu_tuple_cost': 0.5,
                                       'random_page_cost': 0.3}}}},
                          'fakeScope': None}

        # NOTE: For local objects, method will not distinguish
        # local objects or tables, might overwrite the variables
        test_parse = self.test_dbms.parse_dbms_variables(test_dbms_vars)

        self.assertEqual(len(list(test_parse.keys())), 6)
        self.assertEqual(test_parse.get('GlobalView1.cpu_tuple_cost'), [0.01])
        self.assertEqual(test_parse.get('GlobalView1.random_page_cost'), [0.22])
        self.assertEqual(test_parse.get('GlobalView2.cpu_tuple_cost'), [0.05])
        self.assertEqual(test_parse.get('GlobalView2.random_page_cost'), [0.25])
        self.assertEqual(test_parse.get('LocalView1.cpu_tuple_cost'), [0.5])
        self.assertEqual(test_parse.get('LocalView1.random_page_cost'), [0.3])

        test_scope = {'unknownScope': {'GlobalView1':
                                       {'cpu_tuple_cost': 0.01,
                                        'random_page_cost': 0.22},
                                       'GlobalView2':
                                       {'cpu_tuple_cost': 0.05,
                                        'random_page_cost': 0.25}}}

        with self.assertRaises(Exception):
            self.test_dbms.parse_dbms_variables(test_scope)

    @abstractmethod
    def test_parse_dbms_knobs(self):
        pass

    @abstractmethod
    def test_parse_dbms_metrics(self):
        pass

    def test_calculate_change_in_metrics(self):
        self.assertEqual(self.test_dbms.calculate_change_in_metrics({}, {}), {})

    @abstractmethod
    def test_create_knob_configuration(self):
        pass

    def test_format_bool(self):
        mock_other_knob = mock.Mock(spec=KnobCatalog)
        mock_other_knob.unit = KnobUnitType.OTHER

        self.assertEqual(self.test_dbms.format_bool(BooleanType.TRUE, mock_other_knob),
                         self.test_dbms.true_value)
        self.assertEqual(self.test_dbms.format_bool(BooleanType.FALSE, mock_other_knob),
                         self.test_dbms.false_value)

    def test_format_enum(self):
        mock_enum_knob = mock.Mock(spec=KnobCatalog)
        mock_enum_knob.enumvals = 'apple,oranges,cake'

        self.assertEqual(self.test_dbms.format_enum(0, mock_enum_knob), "apple")
        self.assertEqual(self.test_dbms.format_enum(1, mock_enum_knob), "oranges")
        self.assertEqual(self.test_dbms.format_enum(2, mock_enum_knob), "cake")

    def test_format_integer(self):
        mock_other_knob = mock.Mock(spec=KnobCatalog)
        mock_other_knob.unit = KnobUnitType.OTHER

        test_int = [42, -1, 0, 0.5, '1', 42.0, '42.5', 42.7, '']
        test_int_ans = [42, -1, 0, 1, 1, 42, 43, 43, 0]

        for test_int, actual_test_int in zip(test_int, test_int_ans):
            self.assertEqual(
                self.test_dbms.format_integer(test_int, mock_other_knob), actual_test_int)

    def test_format_real(self):
        mock_other_knob = mock.Mock(spec=KnobCatalog)
        mock_other_knob.unit = KnobUnitType.OTHER

        test_real = [42, -1, 0, 0.5, 1, 42.0, 42.5, 42.7]
        test_real_ans = [42.0, -1.0, 0.0, 0.5, 1.0, 42.0, 42.5, 42.7]

        for test_real, actual_test_real in zip(test_real, test_real_ans):
            self.assertEqual(
                self.test_dbms.format_real(test_real, mock_other_knob), actual_test_real)

    def test_format_string(self):
        pass

    def test_format_timestamp(self):
        pass

    def test_format_dbms_knobs(self):
        self.assertEqual(self.test_dbms.format_dbms_knobs({}), {})

        test_exceptions = {'global.FAKE_KNOB': "20"}

        with self.assertRaises(Exception):
            self.test_dbms.format_dbms_knobs(test_exceptions)


class PostgresParserTests(BaseParserTests, TestCase):

    def setUp(self):
        dbms_obj = DBMSCatalog.objects.filter(
            type=DBMSType.POSTGRES, version="9.6").first()
        self.test_dbms = parser._get(dbms_obj.pk)  # pylint: disable=protected-access

        knob_catalog = KnobCatalog.objects.filter(dbms=dbms_obj)
        tunable_knob_catalog = knob_catalog.filter(tunable=True)
        metric_catalog = MetricCatalog.objects.filter(dbms=dbms_obj)
        numeric_metric_catalog = metric_catalog.filter(metric_type__in=MetricType.numeric())

        self.knob_catalog = {k.name: k for k in knob_catalog}
        self.tunable_knob_catalog = {k.name: k for k in tunable_knob_catalog}
        self.metric_catalog = {m.name: m for m in metric_catalog}
        self.numeric_metric_catalog = {m.name: m for m in numeric_metric_catalog}

    def test_convert_dbms_knobs(self):
        super().test_convert_dbms_knobs()

        test_knobs = {'global.wal_sync_method': 'open_sync',  # Enum
                      'global.random_page_cost': 0.22,  # Real
                      'global.archive_command': 'archive',  # String
                      'global.cpu_tuple_cost': 0.55,  # Real
                      'global.force_parallel_mode': 'regress',  # Enum
                      'global.enable_hashjoin': 'on',  # Bool
                      'global.geqo_effort': 5,  # Int
                      'global.wal_buffers': 1024,  # Int
                      'global.FAKE_KNOB': 20}

        test_convert_knobs = self.test_dbms.convert_dbms_knobs(test_knobs)
        self.assertEqual(len(list(test_convert_knobs.keys())), 3)
        self.assertEqual(test_convert_knobs['global.random_page_cost'], 0.22)

        self.assertEqual(test_convert_knobs['global.wal_sync_method'], 2)
        self.assertEqual(test_convert_knobs['global.wal_buffers'], 1024)

        test_except_knobs = {'global.wal_sync_method': '3'}
        with self.assertRaises(Exception):
            self.test_dbms.convert_dbms_knobs(test_except_knobs)

        test_nontune_knobs = {'global.enable_hashjoin': 'on'}
        self.assertEqual(self.test_dbms.convert_dbms_knobs(test_nontune_knobs), {})

    def test_convert_dbms_metrics(self):
        super().test_convert_dbms_metrics()

        target_obj = target_objectives.THROUGHPUT
        target_obj_instance = target_objectives.get_instance(
            self.test_dbms.dbms_id, target_obj)
        txns_counter = target_obj_instance.transactions_counter

        test_metrics = {}
        for met_name in self.numeric_metric_catalog.keys():
            test_metrics[met_name] = 2

        test_metrics[txns_counter] = 10
        test_metrics['pg_FAKE_METRIC'] = 0

        self.assertEqual(test_metrics.get(target_obj), None)

        test_convert_metrics = self.test_dbms.convert_dbms_metrics(test_metrics, 0.1, target_obj)
        for key, metadata in self.numeric_metric_catalog.items():
            if key == txns_counter:
                self.assertEqual(test_convert_metrics[key], 10)
                continue
            if metadata.metric_type == MetricType.COUNTER:
                self.assertEqual(test_convert_metrics[key], 2)
            else:  # MetricType.STATISTICS
                self.assertEqual(test_convert_metrics[key], 2)

        self.assertEqual(test_convert_metrics[target_obj], 100)
        self.assertEqual(test_convert_metrics.get('pg_FAKE_METRIC'), None)

    def test_parse_version_string(self):
        self.assertTrue(self.test_dbms.parse_version_string("9.6.1"), "9.6")
        self.assertTrue(self.test_dbms.parse_version_string("9.6.3"), "9.6")
        self.assertTrue(self.test_dbms.parse_version_string("10.2.1"), "10.2")
        self.assertTrue(self.test_dbms.parse_version_string("0.0.0"), "0.0")

        with self.assertRaises(Exception):
            self.test_dbms.parse_version_string("postgres")

        with self.assertRaises(Exception):
            self.test_dbms.parse_version_string("1.0")

    def test_extract_valid_variables(self):
        num_tunable_knobs = len(self.tunable_knob_catalog)

        test_empty, test_empty_diff = self.test_dbms.extract_valid_variables(
            {}, self.tunable_knob_catalog)
        self.assertEqual(len(test_empty), num_tunable_knobs)
        num_diffs = sum([len(v) for v in test_empty_diff.values()])
        self.assertEqual(num_diffs, num_tunable_knobs)

        test_vars = {'global.wal_sync_method': 'fsync',
                     'global.random_page_cost': 0.22,
                     'global.Wal_buffers': 1024,
                     'global.archive_command': 'archive',
                     'global.GEQO_EFFORT': 5,
                     'global.enable_hashjoin': 'on',
                     'global.cpu_tuple_cost': 0.55,
                     'global.force_parallel_mode': 'regress',
                     'global.FAKE_KNOB': 'fake'}

        tune_extract, tune_diff = self.test_dbms.extract_valid_variables(
            test_vars, self.tunable_knob_catalog)

        self.assertTrue(
            ('global.wal_buffers', 'global.Wal_buffers') in tune_diff['miscapitalized'])
        self.assertTrue('global.GEQO_EFFORT' in tune_diff['extra'])
        self.assertTrue('global.enable_hashjoin' in tune_diff['extra'])
        self.assertTrue('global.deadlock_timeout' in tune_diff['missing'])
        self.assertTrue('global.temp_buffers' in tune_diff['missing'])
        self.assertTrue(tune_extract.get('global.temp_buffers') is not None)
        self.assertTrue(tune_extract.get('global.deadlock_timeout') is not None)

        self.assertEqual(tune_extract.get('global.wal_buffers'), 1024)
        self.assertEqual(tune_extract.get('global.Wal_buffers'), None)

        self.assertEqual(len(tune_extract), num_tunable_knobs)

        nontune_extract, nontune_diff = self.test_dbms.extract_valid_variables(
            test_vars, self.knob_catalog)

        self.assertTrue(
            ('global.wal_buffers', 'global.Wal_buffers') in nontune_diff['miscapitalized'])
        self.assertTrue(
            ('global.geqo_effort', 'global.GEQO_EFFORT') in nontune_diff['miscapitalized'])
        self.assertTrue('global.FAKE_KNOB' in nontune_diff['extra'])
        self.assertTrue('global.lc_ctype' in nontune_diff['missing'])
        self.assertTrue('global.full_page_writes' in nontune_diff['missing'])

        self.assertEqual(nontune_extract.get('global.wal_buffers'), 1024)
        self.assertEqual(nontune_extract.get('global.geqo_effort'), 5)
        self.assertEqual(nontune_extract.get('global.Wal_buffers'), None)
        self.assertEqual(nontune_extract.get('global.GEQO_EFFORT'), None)

    def test_convert_integer(self):
        super().test_convert_integer()

        # Convert Integer
        knob_unit_bytes = KnobUnitType()
        knob_unit_bytes.unit = 1
        knob_unit_time = KnobUnitType()
        knob_unit_time.unit = 2
        knob_unit_other = KnobUnitType()
        knob_unit_other.unit = 3

        self.assertEqual(self.test_dbms.convert_integer('5', knob_unit_other), 5)
        self.assertEqual(self.test_dbms.convert_integer('0', knob_unit_other), 0)
        self.assertEqual(self.test_dbms.convert_integer('0.0', knob_unit_other), 0)
        self.assertEqual(self.test_dbms.convert_integer('0.5', knob_unit_other), 0)

        self.assertEqual(self.test_dbms
                         .convert_integer('5kB', knob_unit_bytes), 5 * 1024)
        self.assertEqual(self.test_dbms
                         .convert_integer('4MB', knob_unit_bytes), 4 * 1024 ** 2)

        self.assertEqual(self.test_dbms.convert_integer('1d', knob_unit_time), 86400000)
        self.assertEqual(self.test_dbms
                         .convert_integer('20h', knob_unit_time), 72000000)
        self.assertEqual(self.test_dbms
                         .convert_integer('10min', knob_unit_time), 600000)
        self.assertEqual(self.test_dbms.convert_integer('1s', knob_unit_time), 1000)
        self.assertEqual(self.test_dbms.convert_integer('5000ms', knob_unit_time), 5000)

        test_exceptions = [('A', knob_unit_other),
                           ('1S', knob_unit_time),
                           ('1mb', knob_unit_bytes)]

        for failure_case, knob_unit in test_exceptions:
            with self.assertRaises(Exception):
                self.test_dbms.convert_integer(failure_case, knob_unit)

    def test_calculate_change_in_metrics(self):
        super().test_calculate_change_in_metrics()

        test_metric_start = {'pg_stat_bgwriter.buffers_alloc': 256,
                             'pg_stat_archiver.last_failed_wal': "today",
                             'pg_stat_archiver.last_failed_time': "2018-01-10 11:24:30",
                             'pg_stat_user_tables.n_tup_upd': 123,
                             'pg_stat_user_tables.relname': "Customers",
                             'pg_stat_user_tables.relid': 2,
                             'pg_stat_user_tables.last_vacuum': "2018-01-09 12:00:00",
                             'pg_stat_database.tup_fetched': 156,
                             'pg_stat_database.datname': "testOttertune",
                             'pg_stat_database.datid': 1,
                             'pg_stat_database.stats_reset': "2018-01-09 13:00:00",
                             'pg_stat_user_indexes.idx_scan': 23,
                             'pg_stat_user_indexes.relname': "Managers",
                             'pg_stat_user_indexes.relid': 20}

        test_metric_end = {'pg_stat_bgwriter.buffers_alloc': 300,
                           'pg_stat_archiver.last_failed_wal': "today",
                           'pg_stat_archiver.last_failed_time': "2018-01-11 11:24:30",
                           'pg_stat_user_tables.n_tup_upd': 150,
                           'pg_stat_user_tables.relname': "Customers",
                           'pg_stat_user_tables.relid': 2,
                           'pg_stat_user_tables.last_vacuum': "2018-01-10 12:00:00",
                           'pg_stat_database.tup_fetched': 260,
                           'pg_stat_database.datname': "testOttertune",
                           'pg_stat_database.datid': 1,
                           'pg_stat_database.stats_reset': "2018-01-10 13:00:00",
                           'pg_stat_user_indexes.idx_scan': 23,
                           'pg_stat_user_indexes.relname': "Managers",
                           'pg_stat_user_indexes.relid': 20}

        test_adj_metrics = self.test_dbms.calculate_change_in_metrics(
            test_metric_start, test_metric_end)

        self.assertEqual(test_adj_metrics['pg_stat_bgwriter.buffers_alloc'], 44)
        self.assertEqual(test_adj_metrics['pg_stat_archiver.last_failed_wal'], "today")
        self.assertEqual(
            test_adj_metrics['pg_stat_archiver.last_failed_time'], "2018-01-11 11:24:30")
        self.assertEqual(test_adj_metrics['pg_stat_user_tables.n_tup_upd'], 27)
        self.assertEqual(test_adj_metrics['pg_stat_user_tables.relname'], "Customers")
        self.assertEqual(test_adj_metrics['pg_stat_user_tables.relid'], 2)  # MetricType.INFO
        self.assertEqual(test_adj_metrics['pg_stat_user_tables.last_vacuum'], "2018-01-10 12:00:00")
        self.assertEqual(test_adj_metrics['pg_stat_database.tup_fetched'], 104)
        self.assertEqual(test_adj_metrics['pg_stat_database.datname'], "testOttertune")
        self.assertEqual(test_adj_metrics['pg_stat_database.datid'], 1)  # MetricType.INFO
        self.assertEqual(test_adj_metrics['pg_stat_database.stats_reset'], "2018-01-10 13:00:00")
        self.assertEqual(test_adj_metrics['pg_stat_user_indexes.idx_scan'], 0)
        self.assertEqual(test_adj_metrics['pg_stat_user_indexes.relid'], 20)  # MetricType.INFO

    def test_create_knob_configuration(self):
        empty_config = self.test_dbms.create_knob_configuration({})
        self.assertEqual(empty_config, {})

        tuning_knobs = {"global.autovacuum": "on",
                        "global.log_planner_stats": "on",
                        "global.cpu_tuple_cost": 0.5,
                        "global.FAKE_KNOB": 20,
                        "pg_stat_archiver.last_failed_wal": "today"}

        test_config = self.test_dbms.create_knob_configuration(tuning_knobs)

        actual_keys = [("autovacuum", "on"),
                       ("log_planner_stats", "on"),
                       ("cpu_tuple_cost", 0.5),
                       ("FAKE_KNOB", 20)]

        self.assertTrue(len(list(test_config.keys())), 4)

        for k, v in actual_keys:
            self.assertEqual(test_config.get(k), v)

    def test_format_integer(self):
        knob_unit_bytes = KnobUnitType()
        knob_unit_bytes.unit = 1
        knob_unit_time = KnobUnitType()
        knob_unit_time.unit = 2
        knob_unit_other = KnobUnitType()
        knob_unit_other.unit = 3

        self.assertEqual(self.test_dbms.format_integer(5, knob_unit_other), 5)
        self.assertEqual(self.test_dbms.format_integer(0, knob_unit_other), 0)
        self.assertEqual(self.test_dbms.format_integer(-1, knob_unit_other), -1)

        self.assertEqual(self.test_dbms.format_integer(5120, knob_unit_bytes), '5kB')
        self.assertEqual(self.test_dbms.format_integer(4194304, knob_unit_bytes), '4MB')
        self.assertEqual(self.test_dbms.format_integer(4194500, knob_unit_bytes), '4MB')

        self.assertEqual(self.test_dbms.format_integer(86400000, knob_unit_time), '1d')
        self.assertEqual(self.test_dbms.format_integer(72000000, knob_unit_time), '20h')
        self.assertEqual(self.test_dbms.format_integer(600000, knob_unit_time), '10min')
        self.assertEqual(self.test_dbms.format_integer(1000, knob_unit_time), '1s')
        self.assertEqual(self.test_dbms.format_integer(500, knob_unit_time), '500ms')

    def test_format_dbms_knobs(self):
        super().test_format_dbms_knobs()

        test_knobs = {'global.wal_sync_method': 2,  # Enum
                      'global.random_page_cost': 0.22,  # Real
                      'global.archive_command': "archive",  # String
                      'global.cpu_tuple_cost': 0.55,  # Real
                      'global.force_parallel_mode': 2,  # Enum
                      'global.enable_hashjoin': BooleanType.TRUE,  # Bool
                      'global.geqo_effort': 5,  # Int
                      'global.wal_buffers': 1024}  # Int

        test_formatted_knobs = self.test_dbms.format_dbms_knobs(test_knobs)

        self.assertEqual(test_formatted_knobs.get('global.wal_sync_method'), 'open_sync')
        self.assertEqual(test_formatted_knobs.get('global.random_page_cost'), 0.22)
        self.assertEqual(test_formatted_knobs.get('global.archive_command'), "archive")
        self.assertEqual(test_formatted_knobs.get('global.cpu_tuple_cost'), 0.55)
        self.assertEqual(test_formatted_knobs.get('global.force_parallel_mode'), 'regress')
        self.assertEqual(test_formatted_knobs.get('global.enable_hashjoin'), 'on')
        self.assertEqual(test_formatted_knobs.get('global.geqo_effort'), 5)
        self.assertEqual(test_formatted_knobs.get('global.wal_buffers'), '1kB')

    def test_parse_helper(self):
        super().test_parse_helper()

        test_view_vars = {'global': {'wal_sync_method': 'open_sync',
                                     'random_page_cost': 0.22},
                          'local': {'FAKE_KNOB': 'FAKE'}}
        valid_vars = {}
        test_scope = 'global'
        test_parse = self.test_dbms.parse_helper(test_scope, valid_vars, test_view_vars)

        self.assertEqual(len(list(test_parse.keys())), 3)
        self.assertEqual(test_parse.get('global.wal_sync_method'), ['open_sync'])
        self.assertEqual(test_parse.get('global.random_page_cost'), [0.22])
        self.assertEqual(test_parse.get('local.FAKE_KNOB'), ['FAKE'])

    def test_parse_dbms_knobs(self):
        test_knobs = {'global': {'global':
                                 {'wal_sync_method': 'fsync',
                                  'random_page_cost': 0.22,
                                  'wal_buffers': 1024,
                                  'archive_command': 'archive',
                                  'geqo_effort': 5,
                                  'enable_hashjoin': 'on',
                                  'cpu_tuple_cost': 0.55,
                                  'force_parallel_mode': 'regress',
                                  'FAKE_KNOB': 'fake'}}}
        num_knobs = len(self.knob_catalog)

        test_parse_dict, test_parse_log = self.test_dbms.parse_dbms_knobs(test_knobs)

        num_log_entries = sum([len(v) for v in test_parse_log.values()])
        self.assertEqual(num_log_entries, num_knobs - 7)
        self.assertTrue('global.FAKE_KNOB' in test_parse_log['extra'])

        self.assertEqual(len(test_parse_dict), num_knobs)
        self.assertEqual(test_parse_dict['global.wal_sync_method'], 'fsync')
        self.assertEqual(test_parse_dict['global.random_page_cost'], 0.22)

    def test_parse_dbms_metrics(self):
        test_metrics = {'global':
                        {'pg_stat_archiver.last_failed_wal': "today",
                         'pg_stat_bgwriter.buffers_alloc': 256,
                         'pg_stat_archiver.last_failed_time': "2018-01-10 11:24:30"},
                        'database':
                        {'pg_stat_database.tup_fetched': 156,
                         'pg_stat_database.datid': 1,
                         'pg_stat_database.datname': "testOttertune",
                         'pg_stat_database.stats_reset': "2018-01-09 13:00:00"},
                        'table':
                        {'pg_stat_user_tables.last_vacuum': "2018-01-09 12:00:00",
                         'pg_stat_user_tables.relid': 20,
                         'pg_stat_user_tables.relname': "Managers",
                         'pg_stat_user_tables.n_tup_upd': 123},
                        'index':
                        {'pg_stat_user_indexes.idx_scan': 23,
                         'pg_stat_user_indexes.relname': "Customers",
                         'pg_stat_user_indexes.relid': 2}}

        # Doesn't support table or index scope
        with self.assertRaises(Exception):
            num_metrics = len(self.metric_catalog)
            test_parse_dict, test_parse_log = self.test_dbms.parse_dbms_metrics(test_metrics)
            self.assertEqual(len(test_parse_dict), num_metrics)
            self.assertEqual(len(test_parse_log), num_metrics - 14)

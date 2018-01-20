
from django.test import TestCase
from django.conf import settings
from website.parser.postgres import PostgresParser, Postgres96Parser
from website.types import BooleanType, VarType, KnobUnitType, LabelStyleType

class BaseParserTest(TestCase):
    def testConverts(self):
        testDBMS = PostgresParser(2)

        # Convert Bool
        BoolParam = VarType()
        testBool = ['on', 'ON', 'OFF', 'off', 'apavlo']
        testBoolAns = [BooleanType.TRUE, BooleanType.TRUE, BooleanType.FALSE,
                        BooleanType.FALSE, BooleanType.FALSE]
        for i in range(len(testBool)):
            self.assertEqual(
                super(PostgresParser, testDBMS)
                .convert_bool(testBool[i], BoolParam), testBoolAns[i])

        # Convert Enum
        # TODO (jackyl): Change tests for 1-hot encoding,
        # however it's implemented
        EnumParam = VarType()
        EnumParam.enumvals = 'apple,oranges,cake'
        EnumParam.name = "Test"

        self.assertEqual(
            super(PostgresParser, testDBMS)
            .convert_enum('apple', EnumParam), 0)
        self.assertEqual(
            super(PostgresParser, testDBMS)
            .convert_enum('oranges', EnumParam), 1)
        self.assertEqual(
            super(PostgresParser, testDBMS)
            .convert_enum('cake', EnumParam), 2)

        with self.assertRaises(Exception):
            super(PostgresParser, testDBMS).convert_enum('jackyl', EnumParam)

        # Convert Integer
        IntParam = VarType()
        testInt = ['42', '-1', '0', '1', '42.0', '42.5', '42.7']
        testIntAns = [42, -1, 0, 1, 42, 42, 42]

        for i in range(len(testInt)):
            self.assertEqual(
                super(PostgresParser, testDBMS)
                .convert_integer(testInt[i], IntParam), testIntAns[i])

        with self.assertRaises(Exception):
            super(PostgresParser, testDBMS).convert_integer('notInt', IntParam)

        # Convert Real
        RealParam = VarType()
        testReal = ['42.0', '42.2', '42.5', '42.7', '-1', '0', '1']
        testRealAns = [42.0, 42.2, 42.5, 42.7, -1.0, 0.0, 1.0]

        for i in range(len(testReal)):
            self.assertEqual(
                super(PostgresParser, testDBMS)
                .convert_real(testReal[i], RealParam), testRealAns[i])

        with self.assertRaises(Exception):
            super(PostgresParser, testDBMS).convert_real('notReal', RealParam)

    def testConvertCatalogs(self):
        # Convert DBMS Knobs
        testDBMS = Postgres96Parser()
        testKnobs = {'global.wal_sync_method'   : 'open_sync', # Enum
                     'global.random_page_cost'  : 0.22, # Real
                     'global.archive_command'   : 'archive', # String
                     'global.cpu_tuple_cost'    : 0.55, # Real
                     'global.force_parallel_mode'   : 'regress', # Enum
                     'global.enable_hashjoin'   : 'on', # Bool
                     'global.geqo_effort'       : 5, # Int
                     'global.wal_buffers'       : 1024, # Int
                     'global.FAKE_KNOB'         : 20}

        testBaseConvertKnobs = super(PostgresParser, PostgresParser(2)).convert_dbms_knobs(testKnobs)
        self.assertEqual(testBaseConvertKnobs, {})

        testConvertKnobs = testDBMS.convert_dbms_knobs(testKnobs)
        # TODO: (jackyl) Test tunable knobs of other vartypes (string, bool, timestamp)
        self.assertEqual(len(testConvertKnobs.keys()), 3)
        self.assertEqual(testConvertKnobs['global.random_page_cost'], 0.22)
        # NOTE: (jackyl) Fix enum when 1-hot encoding implemented
        self.assertEqual(testConvertKnobs['global.wal_sync_method'], 2)
        self.assertEqual(testConvertKnobs['global.wal_buffers'], 1024)

        testExceptKnobs = {'global.wal_sync_method' : '3'}
        with self.assertRaises(Exception):
            testDBMS.convert_dbms_knobs(testExceptKnobs)

        testNontunableKnobs = {'global.enable_hashjoin' : 'on'}
        self.assertEqual(testDBMS.convert_dbms_knobs(testNontunableKnobs), {})

        # TODO: (jackyl) Need exception test for nonexistent var type

        # Convert DBMS Metrics
        testMetrics = {}

        for key, val in testDBMS.numeric_metric_catalog_.iteritems():
            testMetrics[key] = 2
        testMetrics['pg_stat_database.xact_commit'] = 10
        testMetrics['pg_FAKE_METRIC'] = 0

        self.assertEqual(testMetrics.get('throughput_txn_per_sec'), None)

        testConvertMetrics = testDBMS.convert_dbms_metrics(testMetrics, 0.1)
        for (key, val) in testDBMS.numeric_metric_catalog_.iteritems():
            if (key == testDBMS.transactions_counter):
                self.assertEqual(testConvertMetrics[key], 100)
                continue
            self.assertEqual(testConvertMetrics[key], 20)

        self.assertEqual(testConvertMetrics['throughput_txn_per_sec'], 100)
        self.assertEqual(testConvertMetrics.get('pg_FAKE_METRIC'), None)

    def testExtractValidVariables(self):
        testDBMS = Postgres96Parser()
        numTunableKnobs = len(testDBMS.tunable_knob_catalog_.keys())
        numNontunableKnobs = len(testDBMS.knob_catalog_.keys())

        testEmpty, testEmptyDifflog = testDBMS.extract_valid_variables({}, testDBMS.tunable_knob_catalog_)
        self.assertEqual(len(testEmpty.keys()), numTunableKnobs)
        self.assertEqual(len(testEmptyDifflog), numTunableKnobs)

        testVariables = {'global.wal_sync_method'   : 'fsync',
                         'global.random_page_cost'  : 0.22,
                         'global.Wal_buffers'       : 1024,
                         'global.archive_command'   : 'archive',
                         'global.GEQO_EFFORT'       : 5,
                         'global.enable_hashjoin'   : 'on',
                         'global.cpu_tuple_cost'    : 0.55,
                         'global.force_parallel_mode'   : 'regress',
                         'global.FAKE_KNOB'         : 'fake'}

        tuneExtract, tuneDifflog = testDBMS.extract_valid_variables(testVariables, testDBMS.tunable_knob_catalog_)

        self.assertTrue(('miscapitalized', 'global.wal_buffers', 'global.Wal_buffers', 1024) in tuneDifflog)
        self.assertTrue(('extra', None, 'global.GEQO_EFFORT', 5) in tuneDifflog)
        self.assertTrue(('extra', None, 'global.enable_hashjoin', 'on') in tuneDifflog)
        self.assertTrue(('missing', 'global.deadlock_timeout', None, None) in tuneDifflog)
        self.assertTrue(('missing', 'global.temp_buffers', None, None) in tuneDifflog)
        self.assertTrue(tuneExtract.get('global.temp_buffers') != None)
        self.assertTrue(tuneExtract.get('global.deadlock_timeout') != None)

        self.assertEqual(tuneExtract.get('global.wal_buffers'), 1024)
        self.assertEqual(tuneExtract.get('global.Wal_buffers'), None)

        self.assertEqual(len(tuneExtract), len(testDBMS.tunable_knob_catalog_))

        nontuneExtract, nontuneDifflog = testDBMS.extract_valid_variables(testVariables, testDBMS.knob_catalog_)

        self.assertTrue(('miscapitalized', 'global.wal_buffers', 'global.Wal_buffers', 1024) in nontuneDifflog)
        self.assertTrue(('miscapitalized', 'global.geqo_effort', 'global.GEQO_EFFORT', 5) in nontuneDifflog)
        self.assertTrue(('extra', None, 'global.FAKE_KNOB', 'fake') in nontuneDifflog)
        self.assertTrue(('missing', 'global.lc_ctype', None, None) in nontuneDifflog)
        self.assertTrue(('missing', 'global.full_page_writes', None, None) in nontuneDifflog)

        self.assertEqual(nontuneExtract.get('global.wal_buffers'), 1024)
        self.assertEqual(nontuneExtract.get('global.geqo_effort'), 5)
        self.assertEqual(nontuneExtract.get('global.Wal_buffers'), None)
        self.assertEqual(nontuneExtract.get('global.GEQO_EFFORT'), None)

    def testParses(self):
        testDBMS = Postgres96Parser()

        # Parse Helper
        testViewVars = {'global' : {'wal_sync_method' : 'open_sync',
                                    'random_page_cost': 0.22},
                        'local'  : {'FAKE_KNOB' : 'FAKE'}}

        testParse = testDBMS.parse_helper(testViewVars)

        self.assertEqual(len(testParse.keys()), 3)
        self.assertEqual(testParse.get('global.wal_sync_method'), ['open_sync'])
        self.assertEqual(testParse.get('global.random_page_cost'), [0.22])
        self.assertEqual(testParse.get('local.FAKE_KNOB'), ['FAKE'])

        # Parse DBMS Variables
        testDBMSVars = {'global' : {'GlobalView1'   :
                                    {'cpu_tuple_cost'   : 0.01,
                                     'random_page_cost' : 0.22},
                                    'GlobalView2'   :
                                    {'cpu_tuple_cost'   : 0.05,
                                     'random_page_cost' : 0.25}},
                        'local'  : {'CustomerTable' :
                                    {'LocalView1'   :
                                     {'LocalObj1'   :
                                        {'cpu_tuple_cost'   : 0.1,
                                         'random_page_cost' : 0.2},
                                      'LocalObj2'   :
                                        {'cpu_tuple_cost'   : 0.5,
                                         'random_page_cost' : 0.3}}}},
                        'fakeScope' : None}
        # NOTE: For local objects, method will not distinguish local objects or tables.
        testParse = testDBMS.parse_dbms_variables(testDBMSVars)

        self.assertEqual(len(testParse.keys()), 6)
        self.assertEqual(testParse.get('GlobalView1.cpu_tuple_cost'), [0.01])
        self.assertEqual(testParse.get('GlobalView1.random_page_cost'), [0.22])
        self.assertEqual(testParse.get('GlobalView2.cpu_tuple_cost'), [0.05])
        self.assertEqual(testParse.get('GlobalView2.random_page_cost'), [0.25])
        self.assertEqual(testParse.get('LocalView1.cpu_tuple_cost'), [0.5])
        self.assertEqual(testParse.get('LocalView1.random_page_cost'), [0.3])

        testScope = {'unknownScope' : {'GlobalView1'   :
                                        {'cpu_tuple_cost'   : 0.01,
                                         'random_page_cost' : 0.22},
                                       'GlobalView2'   :
                                        {'cpu_tuple_cost'   : 0.05,
                                         'random_page_cost' : 0.25}}}

        with self.assertRaises(Exception):
            testDBMS.parse_dbms_variables(testScope)

        # Parse DBMS Knobs
        testKnobs = {'global' : {'global' :
                                {'wal_sync_method'  : 'fsync',
                                 'random_page_cost' : 0.22,
                                 'wal_buffers'      : 1024,
                                 'archive_command'  :'archive',
                                 'geqo_effort'      : 5,
                                 'enable_hashjoin'  : 'on',
                                 'cpu_tuple_cost'   : 0.55,
                                 'force_parallel_mode'  : 'regress',
                                 'FAKE_KNOB'        : 'fake'}}}


        (testParseDict, testParseLog) = testDBMS.parse_dbms_knobs(testKnobs)

        self.assertEqual(len(testParseLog), len(testDBMS.knob_catalog_.keys()) - 7)
        self.assertTrue(('extra', None, 'global.FAKE_KNOB', 'fake') in testParseLog)

        self.assertEqual(len(testParseDict.keys()), len(testDBMS.knob_catalog_.keys()))
        self.assertEqual(testParseDict['global.wal_sync_method'], 'fsync')
        self.assertEqual(testParseDict['global.random_page_cost'], 0.22)

        # Parse DBMS Metrics
        testMetrics =  {'global'    :
                            {'pg_stat_archiver.last_failed_wal' : "today",
                             'pg_stat_bgwriter.buffers_alloc' : 256,
                             'pg_stat_archiver.last_failed_time' : "2018-01-10 11:24:30"},
                        'database'  :
                            {'pg_stat_database.tup_fetched' : 156,
                             'pg_stat_database.datid'       : 1,
                             'pg_stat_database.datname'     : "testOttertune",
                             'pg_stat_database.stats_reset' : "2018-01-09 13:00:00"},
                        'table'     :
                            {'pg_stat_user_tables.last_vacuum'  : "2018-01-09 12:00:00",
                             'pg_stat_user_tables.relid'        : 20,
                             'pg_stat_user_tables.relname'      : "Managers",
                             'pg_stat_user_tables.n_tup_upd'    : 123},
                        'index'     :
                            {'pg_stat_user_indexes.idx_scan'    : 23,
                             'pg_stat_user_indexes.relname'     : "Customers",
                             'pg_stat_user_indexes.relid'       : 2}}

        # testParseDict, testParseLog = testDBMS.parse_dbms_metrics(testMetrics)
        #
        # self.assertEqual(len(testParseDict.keys()), len(testDBMS.metric_catalog_.keys()))
        # self.assertEqual(len(testParseLog), len(testDBMS.metric_catalog_.keys()) - 14)

    def testCalculateChangeInMetrics(self):
        testDBMS = Postgres96Parser()
        self.assertEqual(testDBMS.calculate_change_in_metrics({}, {}), {})

        testMetricStart =  {'pg_stat_bgwriter.buffers_alloc'    : 256,
                            'pg_stat_archiver.last_failed_wal'  : "today",
                            'pg_stat_archiver.last_failed_time' : "2018-01-10 11:24:30",
                            'pg_stat_user_tables.n_tup_upd'     : 123,
                            'pg_stat_user_tables.relname'       : "Customers",
                            'pg_stat_user_tables.relid'         : 2,
                            'pg_stat_user_tables.last_vacuum'   : "2018-01-09 12:00:00",
                            'pg_stat_database.tup_fetched'      : 156,
                            'pg_stat_database.datname'          : "testOttertune",
                            'pg_stat_database.datid'            : 1,
                            'pg_stat_database.stats_reset'      : "2018-01-09 13:00:00",
                            'pg_stat_user_indexes.idx_scan'     : 23,
                            'pg_stat_user_indexes.relname'      : "Managers",
                            'pg_stat_user_indexes.relid'        : 20}

        testMetricEnd =    {'pg_stat_bgwriter.buffers_alloc'    : 300,
                            'pg_stat_archiver.last_failed_wal'  : "today",
                            'pg_stat_archiver.last_failed_time' : "2018-01-11 11:24:30",
                            'pg_stat_user_tables.n_tup_upd'     : 150,
                            'pg_stat_user_tables.relname'       : "Customers",
                            'pg_stat_user_tables.relid'         : 2,
                            'pg_stat_user_tables.last_vacuum'   : "2018-01-10 12:00:00",
                            'pg_stat_database.tup_fetched'      : 260,
                            'pg_stat_database.datname'          : "testOttertune",
                            'pg_stat_database.datid'            : 1,
                            'pg_stat_database.stats_reset'      : "2018-01-10 13:00:00",
                            'pg_stat_user_indexes.idx_scan'     : 23,
                            'pg_stat_user_indexes.relname'      : "Managers",
                            'pg_stat_user_indexes.relid'        : 20}

        testAdjMetrics = testDBMS.calculate_change_in_metrics(testMetricStart, testMetricEnd)

        self.assertEqual(testAdjMetrics['pg_stat_bgwriter.buffers_alloc'], 44)
        self.assertEqual(testAdjMetrics['pg_stat_archiver.last_failed_wal'], "today")
        self.assertEqual(testAdjMetrics['pg_stat_archiver.last_failed_time'], "2018-01-11 11:24:30")
        self.assertEqual(testAdjMetrics['pg_stat_user_tables.n_tup_upd'], 27)
        self.assertEqual(testAdjMetrics['pg_stat_user_tables.relname'], "Customers")
        self.assertEqual(testAdjMetrics['pg_stat_user_tables.relid'], 0)
        self.assertEqual(testAdjMetrics['pg_stat_user_tables.last_vacuum'], "2018-01-10 12:00:00")
        self.assertEqual(testAdjMetrics['pg_stat_database.tup_fetched'], 104)
        self.assertEqual(testAdjMetrics['pg_stat_database.datname'], "testOttertune")
        self.assertEqual(testAdjMetrics['pg_stat_database.datid'], 0)
        self.assertEqual(testAdjMetrics['pg_stat_database.stats_reset'], "2018-01-10 13:00:00")
        self.assertEqual(testAdjMetrics['pg_stat_user_indexes.idx_scan'], 0)
        self.assertEqual(testAdjMetrics['pg_stat_user_indexes.relid'], 0)

    def testCreateConfig(self):
        import os
        from website.settings import CONFIG_DIR
        testDBMS = Postgres96Parser()

        tuningKnobs = {}
        customKnobs = {}

        categories = set()
        for (k, v) in testDBMS.knob_catalog_.iteritems():
            if (v.category not in categories):
                categories.add(v.category)
                tuningKnobs.update({k : v.default})

        configRes = testDBMS.create_knob_configuration(tuningKnobs, customKnobs)
        config_path = os.path.join(CONFIG_DIR, testDBMS.knob_configuration_filename)
        with open(config_path, 'r') as f:
            configHeader = f.read()

        self.assertTrue(configHeader in configRes)

        for k in tuningKnobs:
            line = k[len('global.'):]+ ' = \'' + str(tuningKnobs[k]) + '\''
            self.assertTrue(line in configRes)

        with self.assertRaises(Exception):
            testDBMS.create_knob_configuration(tuningKnobs, {"FAKE_KNOB" : 1})

    def testGetNondefaultSettings(self):
        testDBMS = Postgres96Parser()

        self.assertEqual(testDBMS.get_nondefault_knob_settings({}), {})

        testNondefaults =  {'global.archive_command'    : "nonempty", # ''
                            'global.geqo_effort'        : "5", # Default 5
                            'global.enable_hashjoin'    : "off", # Default On
                            'global.cpu_tuple_cost'     : "0.01", # Default 0.01
                            'global.force_parallel_mode': "off",
                            'global.wal_sync_method'    : "fsync", # fdatasync
                            'global.random_page_cost'   : "5.0", # Default 4.0
                            'global.wal_buffers'        : "-1"} # Default -1

        testResults = testDBMS.get_nondefault_knob_settings(testNondefaults)

        self.assertEqual(len(testResults.keys()), 2)
        self.assertEqual(testResults.get('global.archive_command'), 'nonempty')
        self.assertEqual(testResults.get('global.enable_hashjoin'), 'off')
        self.assertEqual(testResults.get('global.wal_sync_method'), None)
        self.assertEqual(testResults.get('global.wal_buffers'), None)

    def testFormats(self):
        testDBMS = PostgresParser(2)

        KnobUnitBytes = KnobUnitType()
        KnobUnitBytes.unit = 1
        KnobUnitTime = KnobUnitType()
        KnobUnitTime.unit = 2
        KnobUnitOther = KnobUnitType()
        KnobUnitOther.unit = 3

        # Format Bool
        self.assertEqual(super(PostgresParser, testDBMS)
                         .format_bool(BooleanType.TRUE, KnobUnitOther), 'on')
        self.assertEqual(super(PostgresParser, testDBMS)
                         .format_bool(BooleanType.FALSE, KnobUnitOther), 'off')

        # Format Enum
        EnumParam = VarType()
        EnumParam.enumvals = 'apple,oranges,cake'

        self.assertEqual(super(PostgresParser, testDBMS)
                         .format_enum(0, EnumParam), "apple")
        self.assertEqual(super(PostgresParser, testDBMS)
                         .format_enum(1, EnumParam), "oranges")
        self.assertEqual(super(PostgresParser, testDBMS)
                         .format_enum(2, EnumParam), "cake")

        # Format Integer
        testInt = [42, -1, 0, 0.5, 1, 42.0, 42.5, 42.7]
        testIntAns = [42, -1, 0, 1, 1, 42, 43, 43]

        for i in range(len(testInt)):
            self.assertEqual(
                super(PostgresParser, testDBMS)
                .format_integer(testInt[i], KnobUnitOther), testIntAns[i])

        # Format Real
        testReal = [42, -1, 0, 0.5, 1, 42.0, 42.5, 42.7]
        testRealAns = [42.0, -1.0, 0.0, 0.5, 1.0, 42.0, 42.5, 42.7]

        for i in range(len(testReal)):
            self.assertEqual(
                super(PostgresParser, testDBMS)
                .format_real(testReal[i], KnobUnitOther), testRealAns[i])

    def testFormatKnobs(self):
        # Format DBMS Params
        testDBMS = Postgres96Parser()
        self.assertEqual(testDBMS.format_dbms_knobs({}), {})

        testKnobs = {'global.wal_sync_method'   : 2, # Enum
                     'global.random_page_cost'  : 0.22, # Real
                     #'global.archive_command'   : "archive", # String
                     'global.cpu_tuple_cost'    : 0.55, # Real
                     'global.force_parallel_mode'   : 2, # Enum
                     'global.enable_hashjoin'   : BooleanType.TRUE, # Bool
                     'global.geqo_effort'       : 5, # Int
                     'global.wal_buffers'       : 1024} # Int
                     # 'global.FAKE_KNOB'         : "20"}
        # FIXME: (jackyl) Add tests for string types and timestamps
        testFormattedKnobs = testDBMS.format_dbms_knobs(testKnobs)
        self.assertEqual(testFormattedKnobs.get('global.wal_sync_method'), 'open_sync')
        self.assertEqual(testFormattedKnobs.get('global.random_page_cost'), 0.22)
        self.assertEqual(testFormattedKnobs.get('global.cpu_tuple_cost'), 0.55)
        self.assertEqual(testFormattedKnobs.get('global.force_parallel_mode'), 'regress')
        self.assertEqual(testFormattedKnobs.get('global.enable_hashjoin'), 'on')
        self.assertEqual(testFormattedKnobs.get('global.geqo_effort'), 5)
        self.assertEqual(testFormattedKnobs.get('global.wal_buffers'), '1kB')

    def testFilters(self):
        testDBMS = Postgres96Parser()

        # Filter Numeric Metrics
        testMetrics =  {'pg_stat_bgwriter.checkpoints_req'  : (2L, 'global'),
                        'pg_stat_archiver.last_failed_wal'  : (1L, 'global'),
                        'pg_stat_database.stats_reset'      : (6L, 'database'),
                        'pg_statio_user_indexes.indexrelname'   : (1L, 'index'),
                        'pg_stat_bgwriter.maxwritten_clean' : (2L, 'global'),
                        'pg_stat_database.tup_fetched'      : (2L, 'database'),
                        'pg_statio_user_tables.heap_blks_read'  : (2L, 'table'),
                        'pg_FAKE_METRIC'                    : (2L, 'database')}

        fMetrics = testDBMS.filter_numeric_metrics(testMetrics)

        self.assertEqual(len(fMetrics.keys()), 4)
        self.assertEqual(fMetrics.get('pg_stat_bgwriter.checkpoints_req'),
                        (2L, 'global'))
        self.assertEqual(fMetrics.get('pg_stat_archiver.last_failed_wal'), None)
        self.assertEqual(fMetrics.get('pg_stat_database.stats_reset'), None)
        self.assertEqual(fMetrics.get('pg_statio_user_indexes.indexrelname'),
                        None)
        self.assertEqual(fMetrics.get('pg_stat_bgwriter.maxwritten_clean'),
                        (2L, 'global'))
        self.assertEqual(fMetrics.get('pg_stat_database.tup_fetched'),
                        (2L, 'database'))
        self.assertEqual(fMetrics.get('pg_statio_user_tables.heap_blks_read'),
                        (2L, 'table'))
        self.assertEqual(fMetrics.get('pg_FAKE_KNOB'), None)

        # Filter Tunable Knobs
        testKnobs = {'global.wal_sync_method'   : 5,
                     'global.random_page_cost'  : 3,
                     'global.archive_command'   : 1,
                     'global.cpu_tuple_cost'    : 3,
                     'global.force_parallel_mode'   : 5,
                     'global.enable_hashjoin'   : 3,
                     'global.geqo_effort'       : 2,
                     'global.wal_buffers'       : 2,
                     'global.FAKE_KNOB'         : 2}

        fKnobs = testDBMS.filter_tunable_knobs(testKnobs)

        self.assertEqual(len(fKnobs.keys()), 3)
        self.assertEqual(fKnobs.get('global.wal_sync_method'), 5)
        self.assertEqual(fKnobs.get('global.wal_buffers'), 2)
        self.assertEqual(fKnobs.get('global.random_page_cost'), 3)
        self.assertEqual(fKnobs.get('global.cpu_tuple_cost'), None)
        self.assertEqual(fKnobs.get('global.FAKE_KNOB'), None)

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

        self.assertEqual(testDBMS
                         .knob_configuration_filename, 'postgresql.conf')
        self.assertEqual(testDBMS
                         .transactions_counter, 'pg_stat_database.xact_commit')

    def testParseVersionString(self):
        testDBMS = PostgresParser(2)

        self.assertTrue(testDBMS.parse_version_string("9.6.1"), "9.6")
        self.assertTrue(testDBMS.parse_version_string("9.6.3"), "9.6")
        self.assertTrue(testDBMS.parse_version_string("10.2.1"), "10.2")
        self.assertTrue(testDBMS.parse_version_string("0.0.0"), "0.0")

        with self.assertRaises(Exception):
            testDBMS.parse_version_string("postgres")

        with self.assertRaises(Exception):
            testDBMS.parse_version_string("1.0")

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

        self.assertEqual(testDBMS
                         .convert_integer('5kB', KnobUnitBytes), 5 * 1024)
        self.assertEqual(testDBMS
                         .convert_integer('4MB', KnobUnitBytes), 4 * 1024 ** 2)

        self.assertEqual(testDBMS.convert_integer('1d', KnobUnitTime), 86400000)
        self.assertEqual(testDBMS
                         .convert_integer('20h', KnobUnitTime), 72000000)
        self.assertEqual(testDBMS
                         .convert_integer('10min', KnobUnitTime), 600000)
        self.assertEqual(testDBMS.convert_integer('1s', KnobUnitTime), 1000)

        testExceptions = [('A', KnobUnitOther),
                          ('', KnobUnitOther),
                          ('-20', KnobUnitOther),
                          ('-1s', KnobUnitTime),
                          ('1S', KnobUnitTime),
                          ('-1MB', KnobUnitBytes),
                          ('1mb', KnobUnitBytes)]

        for failCase, KnobUnit in testExceptions:
            with self.assertRaises(Exception):
                testDBMS.convert_integer(failCase, KnobUnit)

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
        self.assertEqual(testDBMS.format_integer(1000, KnobUnitTime), '1s')
        self.assertEqual(testDBMS.format_integer(500, KnobUnitTime), '500ms')

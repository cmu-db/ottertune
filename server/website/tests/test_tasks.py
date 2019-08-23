#
# OtterTune - test_tasks.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import copy
import numpy as np
from django.test import TestCase, override_settings
from django.db import transaction
from website.models import (Workload, PipelineRun, PipelineData,
                            Result, Session, DBMSCatalog, Hardware)
from website.tasks.periodic_tasks import (run_background_tasks,
                                          aggregate_data,
                                          run_workload_characterization,
                                          run_knob_identification)
from website.types import PipelineTaskType, WorkloadStatusType

CELERY_TEST_RUNNER = 'djcelery.contrib.test_runner.CeleryTestSuiteRunner'


@override_settings(CELERY_ALWAYS_EAGER=True, TEST_RUNNER=CELERY_TEST_RUNNER)
class BackgroundTestCase(TestCase):

    fixtures = ['test_website.json']
    serialized_rollback = True

    def testNoError(self):
        result = run_background_tasks.delay()
        self.assertTrue(result.successful())

    def testProcessedWorkloadStatus(self):
        before_workloads = Workload.objects.filter(status=WorkloadStatusType.MODIFIED)
        run_background_tasks.delay()
        for w in before_workloads:
            self.assertEqual(w.status, WorkloadStatusType.PROCESSED)

    def testNoModifiedWorkload(self):
        # First Execution of Modified Workloads
        run_background_tasks.delay()
        first_pipeline_run = PipelineRun.objects.get_latest()
        # Second Execution with no modified workloads
        run_background_tasks.delay()
        second_pipeline_run = PipelineRun.objects.get_latest()
        # Check that the BG task has not run
        self.assertEqual(first_pipeline_run.start_time, second_pipeline_run.start_time)

    # Test that an empty workload is ignored by the BG task
    def testEmptyWorkload(self):
        with transaction.atomic():
            # Create empty workload
            empty_workload = Workload.objects.create_workload(dbms=DBMSCatalog.objects.get(pk=1),
                                                              hardware=Hardware.objects.get(pk=1),
                                                              name="empty_workload")

            result = run_background_tasks.delay()
        # Check that BG task successfully finished
        self.assertTrue(result.successful())
        # Check that the empty workload is still in MODIFIED Status
        self.assertEqual(empty_workload.status, 1)
        pipeline_data = PipelineData.objects.filter(pipeline_run=PipelineRun.objects.get_latest())
        # Check that the empty workload is not in the pipeline datas
        self.assertNotIn(empty_workload.pk, pipeline_data.values_list('workload_id', flat=True))

    # Test that a workload that contain only one knob configuration will be ignored by the BG task
    def testUniqueKnobConfigurationWorkload(self):
        # Get workload to copy data from
        origin_workload = Workload.objects.get(pk=1)
        origin_session = Session.objects.get(pk=1)
        # Create empty workload
        fix_workload = Workload.objects.create_workload(dbms=origin_workload.dbms,
                                                        hardware=origin_workload.hardware,
                                                        name="fixed_knob_workload")

        fix_knob_data = Result.objects.filter(workload=origin_workload,
                                              session=origin_session)[0].knob_data
        # Add 5 Result with the same Knob Configuration
        for res in Result.objects.filter(workload=origin_workload, session=origin_session)[:4]:
            Result.objects.create_result(res.session, res.dbms, fix_workload,
                                         fix_knob_data, res.metric_data,
                                         res.observation_start_time,
                                         res.observation_end_time,
                                         res.observation_time)

        result = run_background_tasks.delay()
        # Check that BG task successfully finished
        self.assertTrue(result.successful())
        # Check that the empty workload is still in MODIFIED Status
        self.assertEqual(fix_workload.status, 1)
        pipeline_data = PipelineData.objects.filter(pipeline_run=PipelineRun.objects.get_latest())
        # Check that the empty workload is not in the pipeline datas
        self.assertNotIn(fix_workload.pk, pipeline_data.values_list('workload_id', flat=True))

    def testNoWorkloads(self):
        # delete any existing workloads
        workloads = Workload.objects.all()
        workloads.delete()

        # background task should not fail
        result = run_background_tasks.delay()
        self.assertTrue(result.successful())

    def testNewPipelineRun(self):
        # this test currently relies on the fixture data so that
        # it actually tests anything
        workloads = Workload.objects.all()
        if len(workloads) > 0:
            runs_before = len(PipelineRun.objects.all())
            run_background_tasks.delay()
            runs_after = len(PipelineRun.objects.all())
            self.assertEqual(runs_before + 1, runs_after)

    def checkNewTask(self, task_type):
        workloads = Workload.objects.all()
        pruned_before = [len(PipelineData.objects.filter(
            workload=workload, task_type=task_type)) for workload in workloads]
        run_background_tasks.delay()
        pruned_after = [len(PipelineData.objects.filter(
            workload=workload, task_type=task_type)) for workload in workloads]
        for before, after in zip(pruned_before, pruned_after):
            self.assertEqual(before + 1, after)

    def testNewPrunedMetrics(self):
        self.checkNewTask(PipelineTaskType.PRUNED_METRICS)

    def testNewRankedKnobs(self):
        self.checkNewTask(PipelineTaskType.RANKED_KNOBS)


class AggregateTestCase(TestCase):

    fixtures = ['test_website.json']

    def testValidWorkload(self):
        workloads = Workload.objects.all()
        valid_workload = workloads[0]
        wkld_results = Result.objects.filter(workload=valid_workload)
        dicts = aggregate_data(wkld_results)
        keys = ['data', 'rowlabels', 'columnlabels']
        for d in dicts:
            for k in keys:
                self.assertIn(k, d)


class PrunedMetricTestCase(TestCase):

    fixtures = ['test_website.json']

    def testValidPrunedMetrics(self):
        workloads = Workload.objects.all()
        wkld_results = Result.objects.filter(workload=workloads[0])
        metric_data = aggregate_data(wkld_results)[1]
        pruned_metrics = run_workload_characterization(metric_data)
        for m in pruned_metrics:
            self.assertIn(m, metric_data['columnlabels'])


class RankedKnobTestCase(TestCase):

    fixtures = ['test_website.json']

    def testValidImportantKnobs(self):
        workloads = Workload.objects.all()
        wkld_results = Result.objects.filter(workload=workloads[0])
        knob_data, metric_data = aggregate_data(wkld_results)

        # instead of doing actual metric pruning by factor analysis / clustering,
        # just randomly select 5 nonconstant metrics
        nonconst_metric_columnlabels = []
        for col, cl in zip(metric_data['data'].T, metric_data['columnlabels']):
            if np.any(col != col[0]):
                nonconst_metric_columnlabels.append(cl)

        num_metrics = min(5, len(nonconst_metric_columnlabels))
        selected_columnlabels = np.random.choice(nonconst_metric_columnlabels,
                                                 num_metrics, replace=False)
        pruned_metric_idxs = [i for i, metric_name in
                              enumerate(metric_data['columnlabels'])
                              if metric_name in selected_columnlabels]
        pruned_metric_data = {
            'data': metric_data['data'][:, pruned_metric_idxs],
            'rowlabels': copy.deepcopy(metric_data['rowlabels']),
            'columnlabels': [metric_data['columnlabels'][i] for i in pruned_metric_idxs]
        }

        # run knob_identification using knob_data and fake pruned metrics
        ranked_knobs = run_knob_identification(knob_data, pruned_metric_data,
                                               workloads[0].dbms)
        for k in ranked_knobs:
            self.assertIn(k, knob_data['columnlabels'])

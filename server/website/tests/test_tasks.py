# this will be for unit-testing tasks
from django.test import TestCase

from website.tasks import periodic_tasks
from website.models import Workload, PipelineRun

class BackgroundTestCase(TestCase):

    fixtures = ['test_website.json']

    def testNoError(self):
        result = periodic_tasks.run_background_tasks.delay()
        self.assertTrue(result.successful())

    def testNoWorkloads(self):
        # delete any existing workloads
        workloads = Workload.objects.all()
        workloads.delete()

        # background task should not fail
        result = periodic_tasks.run_background_tasks.delay()
        self.assertTrue(result.successful())

    def testNewPipelineRun(self):
        # this test currently relies on the fixture data so that it actually tests anything
        workloads = Workload.objects.all()
        if len(workloads) > 0:
            runs_before = len(PipelineRun.objects.all())
            periodic_tasks.run_background_tasks.delay()
            runs_after = len(PipelineRun.objects.all())
            self.assertEqual(runs_before + 1, runs_after)

        # does it make sense to test that for each workload, there exists corresponding knob and metric data for the most recent pipelineRun?

class AggregateTestCase(TestCase):

    fixtures = ['test_website.json']

    def testValidWorkload(self):
        workloads = Workload.objects.all()
        valid_workload = workloads[0]
        dicts = periodic_tasks.aggregate_data(valid_workload)
        keys = ['data', 'rowlabels', 'columnlabels']
        for d in dicts:
            for k in keys:
                self.assertIn(k, d)

class MetricKnobTestCase(TestCase):

    fixtures = ['test_website.json']

    @classmethod
    def setUpClass(cls):
        super(MetricKnobTestCase, cls).setUpClass()
        workloads = Workload.objects.all()
        valid_workload = workloads[0]
        cls.knob_data, cls.metric_data = periodic_tasks.aggregate_data(valid_workload)

    def testValidPrunedMetrics(self):
        pruned_metrics = periodic_tasks.run_workload_characterization(self.metric_data)
        for m in pruned_metrics:
            self.assertIn(m, self.metric_data['columnlabels'])

    def testValidImportantKnobs(self):
        # uses unpruned metrics, is this an issue?
        ranked_knobs = periodic_tasks.run_knob_identification(self.knob_data, self.metric_data)
        for k in ranked_knobs:
            self.assertIn(k, self.knob_data['columnlabels'])

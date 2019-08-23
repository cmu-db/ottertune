#
# OtterTune - test_upload.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import os

from django.core.urlresolvers import reverse
from django.test import TestCase

from website.models import Result, Workload
from website.settings import PROJECT_ROOT

from .utils import (TEST_BASIC_SESSION_ID, TEST_BASIC_SESSION_UPLOAD_CODE,
                    TEST_PASSWORD, TEST_TUNING_SESSION_ID, TEST_TUNING_SESSION_UPLOAD_CODE,
                    TEST_USERNAME, TEST_WORKLOAD_ID)


class UploadResultsTests(TestCase):

    fixtures = ['test_website.json']

    def setUp(self):
        self.client.login(username=TEST_USERNAME, password=TEST_PASSWORD)
        test_files_dir = os.path.join(PROJECT_ROOT, 'tests', 'test_files')
        self.upload_files = {
            'metrics_before': os.path.join(test_files_dir, 'sample_metrics_start.json'),
            'metrics_after': os.path.join(test_files_dir, 'sample_metrics_end.json'),
            'knobs': os.path.join(test_files_dir, 'sample_knobs.json'),
            'summary': os.path.join(test_files_dir, 'sample_summary.json')
        }

    @staticmethod
    def open_files(file_info):
        files = {}
        for name, path in list(file_info.items()):
            files[name] = open(path)
        return files

    @staticmethod
    def close_files(files):
        for name, fp in list(files.items()):
            if name != 'upload_code':
                fp.close()

    def upload_to_session_ok(self, session_id, upload_code):
        num_initial_results = Result.objects.filter(session__id=session_id).count()
        form_addr = reverse('new_result')
        post_data = self.open_files(self.upload_files)
        post_data['upload_code'] = upload_code
        response = self.client.post(form_addr, post_data)
        self.close_files(post_data)
        self.assertEqual(response.status_code, 200)
        num_final_results = Result.objects.filter(session__id=session_id).count()
        self.assertEqual(num_final_results - num_initial_results, 1)

    def upload_to_session_fail_invalidation(self, session_id, upload_code):
        form_addr = reverse('new_result')
        post_data = {'upload_code': upload_code}
        response = self.client.post(form_addr, post_data)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "New result form is not valid:")
        self.assertContains(response, "This field is required", 4)

    def upload_to_session_invalid_upload_code(self, session_id):
        form_addr = reverse('new_result')
        post_data = self.open_files(self.upload_files)
        post_data['upload_code'] = "invalid_upload_code"
        response = self.client.post(form_addr, post_data)
        self.close_files(post_data)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Invalid upload code")

    def test_upload_form_not_post(self):
        form_addr = reverse('new_result')
        response = self.client.get(form_addr)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Request type was not POST")

    def test_set_modified_workload_on_upload(self):
        workload0 = Workload.objects.get(pk=TEST_WORKLOAD_ID)
        workload0.status = 3
        workload0.save()
        self.upload_to_session_ok(TEST_BASIC_SESSION_ID, TEST_BASIC_SESSION_UPLOAD_CODE)
        status = Workload.objects.get(pk=TEST_WORKLOAD_ID).status
        self.assertEqual(status, 1)

    def test_upload_to_basic_session_ok(self):
        self.upload_to_session_ok(TEST_BASIC_SESSION_ID, TEST_BASIC_SESSION_UPLOAD_CODE)

    def test_upload_to_tuning_session_ok(self):
        self.upload_to_session_ok(TEST_TUNING_SESSION_ID, TEST_TUNING_SESSION_UPLOAD_CODE)

    def test_upload_to_basic_session_fail_invalidation(self):
        self.upload_to_session_fail_invalidation(TEST_BASIC_SESSION_ID,
                                                 TEST_BASIC_SESSION_UPLOAD_CODE)

    def test_upload_to_tuning_session_fail_invalidation(self):
        self.upload_to_session_fail_invalidation(TEST_TUNING_SESSION_ID,
                                                 TEST_TUNING_SESSION_UPLOAD_CODE)

    def test_upload_to_basic_session_invalid_upload_code(self):
        self.upload_to_session_invalid_upload_code(TEST_BASIC_SESSION_ID)

    def test_upload_to_tuning_session_invalid_upload_code(self):
        self.upload_to_session_invalid_upload_code(TEST_TUNING_SESSION_ID)

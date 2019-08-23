#
# OtterTune - test_views.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Dec 13, 2017

@author: dvanaken
'''

from django.contrib.auth import get_user
from django.core.urlresolvers import reverse
from django.test import TestCase

from .utils import (TEST_BASIC_SESSION_ID, TEST_PASSWORD, TEST_PROJECT_ID, TEST_USERNAME)


class UserAuthViewTests(TestCase):

    fixtures = ['test_user.json', 'test_user_sessions.json']

    def setUp(self):
        pass

    def test_valid_login(self):
        data = {
            'username': TEST_USERNAME,
            'password': TEST_PASSWORD
        }
        response = self.client.post(reverse('login'), data=data)
        self.assertRedirects(response, reverse('home_projects'))
        user = get_user(self.client)
        self.assertTrue(user.is_authenticated())

    def test_invalid_login(self):
        data = {
            'username': 'invalid_user',
            'password': 'invalid_password'
        }
        response = self.client.post(reverse('login'), data=data)
        self.assertEqual(response.status_code, 200)
        user = get_user(self.client)
        self.assertFalse(user.is_authenticated())

    def test_login_view(self):
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)

    def test_new_signup(self):
        response = self.client.get(reverse('signup'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Create Your Account")

    def test_logout_view(self):
        self.client.logout()
        user = get_user(self.client)
        self.assertFalse(user.is_authenticated())


class ProjectViewsTests(TestCase):

    fixtures = ['test_website.json']

    def setUp(self):
        self.client.login(username=TEST_USERNAME, password=TEST_PASSWORD)

    def test_new_project_form(self):
        response = self.client.get(reverse('new_project'))
        self.assertEqual(response.status_code, 200)

    def test_create_project_fail_invalidation(self):
        form_addr = reverse('new_project')
        post_data = {}
        response = self.client.post(form_addr, post_data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'name', "This field is required.")

    def test_create_project_ok(self):
        form_addr = reverse('new_project')
        post_data = {
            'name': 'test_create_project',
            'description': 'testing create project...'
        }
        response = self.client.post(form_addr, post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        project_id = response.context['project'].pk
        self.assertRedirects(response, reverse('project_sessions',
                                               kwargs={'project_id': project_id}))

    def test_edit_project_fail_invalidation(self):
        form_addr = reverse('edit_project', kwargs={'project_id': TEST_PROJECT_ID})
        post_data = {}
        response = self.client.post(form_addr, post_data)
        self.assertFormError(response, 'form', 'name', "This field is required.")

    def test_edit_project_ok(self):
        form_addr = reverse('edit_project', kwargs={'project_id': TEST_PROJECT_ID})
        post_data = {'name': 'new_project_name'}
        response = self.client.post(form_addr, post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertRedirects(response, reverse('project_sessions',
                                               kwargs={'project_id': TEST_PROJECT_ID}))

    def test_delete_zero_project(self):
        form_addr = reverse('delete_project')
        post_data = {'projects': []}
        response = self.client.post(form_addr, post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertRedirects(response, reverse('home_projects'))

    def test_delete_one_project(self):
        form_addr = reverse('delete_project')
        post_data = {'projects': [TEST_PROJECT_ID]}
        response = self.client.post(form_addr, post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertRedirects(response, reverse('home_projects'))

    def test_delete_multiple_projects(self):
        create_form_addr = reverse('new_project')
        project_ids = []
        for i in range(5):
            post_data = {
                'name': 'project_{}'.format(i),
                'description': ""
            }
            response = self.client.post(create_form_addr, post_data, follow=True)
            self.assertEqual(response.status_code, 200)
            project_ids.append(response.context['project'].pk)
        delete_form_addr = reverse('delete_project')
        post_data = {'projects': project_ids}
        response = self.client.post(delete_form_addr, post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertRedirects(response, reverse('home_projects'))


class SessionViewsTests(TestCase):

    fixtures = ['test_website.json']

    def setUp(self):
        self.client.login(username=TEST_USERNAME, password=TEST_PASSWORD)

    def test_new_session_form(self):
        response = self.client.get(reverse('new_session', kwargs={'project_id': TEST_PROJECT_ID}))
        self.assertEqual(response.status_code, 200)

    def test_create_session_fail_invalidation(self):
        form_addr = reverse('new_session', kwargs={'project_id': TEST_PROJECT_ID})
        post_data = {}
        response = self.client.post(form_addr, post_data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response, 'form', 'name', "This field is required.")

    def test_create_basic_session_ok(self):
        form_addr = reverse('new_session', kwargs={'project_id': TEST_PROJECT_ID})
        post_data = {
            'name': 'test_create_basic_session',
            'description': 'testing create basic session...',
            'tuning_session': 'no_tuning_session',
            'cpu': '2',
            'memory': '16.0',
            'storage': '32',
            'dbms': 1
        }
        response = self.client.post(form_addr, post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        session_id = response.context['session'].pk
        self.assertRedirects(response, reverse('session',
                                               kwargs={'project_id': TEST_PROJECT_ID,
                                                       'session_id': session_id}))

    def test_create_tuning_session_ok(self):
        form_addr = reverse('new_session', kwargs={'project_id': TEST_PROJECT_ID})
        post_data = {
            'name': 'test_create_basic_session',
            'description': 'testing create basic session...',
            'tuning_session': 'tuning_session',
            'cpu': '2',
            'memory': '16.0',
            'storage': '32',
            'dbms': 1,
            'target_objective': 'throughput_txn_per_sec'
        }
        response = self.client.post(form_addr, post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        session_id = response.context['session'].pk
        self.assertRedirects(response, reverse('session',
                                               kwargs={'project_id': TEST_PROJECT_ID,
                                                       'session_id': session_id}))

    def test_edit_session_fail_invalidation(self):
        form_addr = reverse('edit_session', kwargs={'project_id': TEST_PROJECT_ID,
                                                    'session_id': TEST_BASIC_SESSION_ID})
        post_data = {}
        response = self.client.post(form_addr, post_data)
        self.assertFormError(response, 'form', 'name', "This field is required.")

    def test_edit_basic_session_ok(self):
        form_addr = reverse('edit_session', kwargs={'project_id': TEST_PROJECT_ID,
                                                    'session_id': TEST_BASIC_SESSION_ID})
        post_data = {
            'name': 'new_session_name',
            'description': 'testing edit basic session...',
            'tuning_session': 'tuning_session',
            'cpu': '2',
            'memory': '16.0',
            'storage': '32',
            'dbms': 1,
            'target_objective': 'throughput_txn_per_sec'
        }
        response = self.client.post(form_addr, post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertRedirects(response, reverse('session',
                                               kwargs={'project_id': TEST_PROJECT_ID,
                                                       'session_id': TEST_BASIC_SESSION_ID}))

    def test_edit_all_knobs_ok(self):
        response = self.client.get(reverse('edit_knobs',
                                           kwargs={'project_id': TEST_PROJECT_ID,
                                                   'session_id': TEST_BASIC_SESSION_ID}))
        self.assertEqual(response.status_code, 200)

    def test_edit_knob_ok(self):
        form_addr = reverse('edit_knobs', kwargs={'project_id': TEST_PROJECT_ID,
                                                  'session_id': TEST_BASIC_SESSION_ID})
        post_data = {
            'name': 'global.wal_writer_delay',
            'minval': '1',
            'maxval': '1000',
            'tunable': 'on'
        }
        response = self.client.post(form_addr, post_data, follow=True)
        self.assertEqual(response.status_code, 204)

    def test_delete_zero_sessions(self):
        form_addr = reverse('delete_session', kwargs={'project_id': TEST_PROJECT_ID})
        post_data = {'sessions': []}
        response = self.client.post(form_addr, post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertRedirects(response, reverse('project_sessions',
                                               kwargs={'project_id': TEST_PROJECT_ID}))

    def test_delete_one_session(self):
        form_addr = reverse('delete_session', kwargs={'project_id': TEST_PROJECT_ID})
        post_data = {'sessions': [TEST_BASIC_SESSION_ID]}
        response = self.client.post(form_addr, post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertRedirects(response, reverse('project_sessions',
                                               kwargs={'project_id': TEST_PROJECT_ID}))

    def test_delete_multiple_sessions(self):
        create_form_addr = reverse('new_session', kwargs={'project_id': TEST_PROJECT_ID})
        session_ids = []
        for i in range(5):
            post_data = {
                'name': 'session_{}'.format(i),
                'description': "",
                'tuning_session': 'no_tuning_session',
                'cpu': '2',
                'memory': '16.0',
                'storage': '32',
                'dbms': 1,
                'target_objective': 'throughput_txn_per_sec'
            }
            response = self.client.post(create_form_addr, post_data, follow=True)
            self.assertEqual(response.status_code, 200)
            session_ids.append(response.context['session'].pk)
        delete_form_addr = reverse('delete_session', kwargs={'project_id': TEST_PROJECT_ID})
        post_data = {'sessions': session_ids}
        response = self.client.post(delete_form_addr, post_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertRedirects(response, reverse('project_sessions',
                                               kwargs={'project_id': TEST_PROJECT_ID}))

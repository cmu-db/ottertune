#
# OtterTune - runner.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Jan 29, 2018

@author: dvanaken
'''

import logging

from django.test.runner import DiscoverRunner


class BaseRunner(DiscoverRunner):

    def run_tests(self, test_labels, extra_tests=None, **kwargs):
        # Disable logging while running tests
        logging.disable(logging.CRITICAL)
        return super(BaseRunner, self).run_tests(test_labels, extra_tests, **kwargs)

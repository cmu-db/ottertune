#
# OtterTune - util_functions.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Aug 14, 2017

@author: dvanaken
'''

from django import template

register = template.Library()  # pylint: disable=invalid-name


@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)


@register.filter
def get_attr(instance, attr_name):
    return getattr(instance, attr_name)


@register.filter
def keys(dictionary):
    return list(dictionary.keys())


@register.filter
def safe_floatformat(text, arg=-2):
    val = template.defaultfilters.floatformat(text, arg)
    return val if val != '' else text

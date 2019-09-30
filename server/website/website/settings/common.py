#
# OtterTune - common.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
"""
Common Django settings for the OtterTune project.

"""

import os
from os.path import abspath, dirname, exists, join
import sys

import djcelery

# ==============================================
# PATH CONFIGURATION
# ==============================================

# Absolute path to this Django project directory.
PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))

# Directory holding all uploaded and intermediate result files
DATA_ROOT = join(PROJECT_ROOT, 'data')

# Absolute path to directory where all oltpbench data is uploaded
UPLOAD_DIR = join(DATA_ROOT, 'media')

# Path to the base DBMS configuration files
CONFIG_DIR = join(PROJECT_ROOT, 'config')

# Where the log files are stored
LOG_DIR = join(PROJECT_ROOT, 'log')

# File/directory upload permissions
FILE_UPLOAD_DIRECTORY_PERMISSIONS = 0o664
FILE_UPLOAD_PERMISSIONS = 0o664

# Path to OtterTune's website and ML modules
OTTERTUNE_LIBS = dirname(PROJECT_ROOT)

# ==============================================
# Path setup
# ==============================================

# Add OtterTune's ML modules to path
sys.path.insert(0, OTTERTUNE_LIBS)

# Try to create the log directory
try:
    if not exists(LOG_DIR):
        os.mkdir(LOG_DIR)
except OSError:  # Invalid permissions
    pass

# ==============================================
# DEBUG CONFIGURATION
# ==============================================

DEBUG = False
TEST_RUNNER = 'tests.runner.BaseRunner'
INTERNAL_IPS = ['127.0.0.1']

# ==============================================
# GENERAL CONFIGURATION
# ==============================================

# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# although not all choices may be available on all operating systems.
# In a Windows environment this must be set to your system time zone.
TIME_ZONE = 'America/New_York'

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = 'en-us'

SITE_ID = 1

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True

# If you set this to False, Django will not format dates, numbers and
# calendars according to the current locale.
USE_L10N = True

# If you set this to False, Django will not use timezone-aware datetimes.
USE_TZ = True

# ==============================================
# MEDIA CONFIGURATION
# ==============================================

# Absolute filesystem path to the directory that will hold user-uploaded files.
# Example: "/var/www/example.com/media/"
MEDIA_ROOT = join(DATA_ROOT, 'media')
MEDIA_ROOT_URL = '/media/'

# URL that handles the media served from MEDIA_ROOT. Make sure to use a
# trailing slash.
# Examples: "http://example.com/media/", "http://media.example.com/"
MEDIA_URL = '/media/'

# ==============================================
# STATIC FILE CONFIGURATION
# ==============================================

# Absolute path to the directory static files should be collected to.
# Don't put anything in this directory yourself; store your static files
# in apps' "static/" subdirectories and in STATICFILES_DIRS.
# Example: "/var/www/example.com/static/"
STATIC_ROOT = join(PROJECT_ROOT, 'website', 'static')

# URL prefix for static files.
# Example: "http://example.com/static/", "http://static.example.com/"
STATIC_URL = '/static/'

# Additional locations of static files
STATICFILES_DIRS = (
    # Put strings here, like "/home/html/static" or "C:/www/django/static".
    # Always use forward slashes, even on Windows.
    # Don't forget to use absolute paths, not relative paths.
)

# List of finder classes that know how to find static files in
# various locations.
STATICFILES_FINDERS = (
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
)

# ==============================================
# TEMPLATE CONFIGURATION
# ==============================================

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            # TEMPLATE_DIRS (use absolute paths)
            join(PROJECT_ROOT, 'website', 'template')
        ],
        'OPTIONS': {
            'context_processors': [
                'django.contrib.auth.context_processors.auth',
                'django.template.context_processors.debug',
                'django.template.context_processors.i18n',
                'django.template.context_processors.media',
                'django.template.context_processors.static',
                'django.template.context_processors.tz',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.csrf',
            ],
            'loaders': [
                'django.template.loaders.filesystem.Loader',
                'django.template.loaders.app_directories.Loader',
            ],
            'debug': DEBUG,
        },
    },
]

# ==============================================
# MIDDLEWARE CONFIGURATION
# ==============================================

MIDDLEWARE_CLASSES = (
    'debug_toolbar.middleware.DebugToolbarMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'request_logging.middleware.LoggingMiddleware',
)

# ==============================================
# APP CONFIGURATION
# ==============================================

INSTALLED_APPS = (
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.sites',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'debug_toolbar',
    # 'django_extensions',
    'django.contrib.admin',
    # Uncomment the next line to enable admin documentation:
    # 'django.contrib.admindocs',
    # 'rest_framework',
    # 'south',
    'djcelery',
    # 'django_celery_beat',
    'website',
)

# ==============================================
# RABBITMQ/CELERY CONFIGURATION
# ==============================================

# Broker URL for RabbitMq
BROKER_URL = 'amqp://guest:guest@localhost:5672//'

# Enable finer-grained reporting: will report 'started' when
# task is executed by a worker.
CELERY_TRACK_STARTED = True

# Worker will execute at most this many tasks before it's killed
# and replaced with a new worker. This helps with memory leaks.
CELERYD_MAX_TASKS_PER_CHILD = 50

# Number of concurrent workers.
CELERYD_CONCURRENCY = 8

djcelery.setup_loader()

# ==============================================
# LOGGING CONFIGURATION
# ==============================================

# A website logging configuration. The only tangible logging
# performed by this configuration is to send an email to
# the site admins on every HTTP 500 error when DEBUG=False.
# See http://docs.djangoproject.com/en/dev/topics/logging for
# more details on how to customize your logging configuration.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
            'datefmt': "%d/%b/%Y %H:%M:%S"
        },
    },
    'handlers': {
        'logfile': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': join(LOG_DIR, 'website.log'),
            'maxBytes': 50000,
            'backupCount': 2,
            'formatter': 'standard',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'mail_admins': {
            'level': 'ERROR',
            'filters': ['require_debug_false'],
            'class': 'django.utils.log.AdminEmailHandler'
        },
    },
    'filters': {
        'require_debug_false': {
            '()': 'django.utils.log.RequireDebugFalse'
        }
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'logfile'],
            'propagate': True,
            'level': 'WARN',
        },
        'django.db.backends': {
            'handlers': ['console', 'logfile'],
            'level': 'WARN',
            'propagate': False,
        },
        'website': {
            'handlers': ['console', 'logfile'],
            'propagate': False,
            'level': 'DEBUG',
        },
        'django.request': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
        # Uncomment to email admins after encountering an error (and debug=False)
        # 'django.request': {
        #     'handlers': ['mail_admins'],
        #     'level': 'ERROR',
        #     'propagate': True,
        # },
    }
}

# ==============================================
# URL CONFIGURATION
# ==============================================

ROOT_URLCONF = 'website.urls'

# ==============================================
# WSGI CONFIGURATION
# ==============================================

# Python dotted path to the WSGI application used by Django's runserver.
WSGI_APPLICATION = 'website.wsgi.application'

# ==============================================
# PASSWORD VALIDATORS
# ==============================================

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 6,
        }
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# ==============================================
# MISC
# ==============================================

SESSION_SERIALIZER = 'django.contrib.sessions.serializers.JSONSerializer'


# Import and override defaults with custom configuration options
# pylint: disable=wildcard-import,wrong-import-position,unused-wildcard-import
from .credentials import *  # pycodestyle: disable=E402
from .constants import *  # pycodestyle: disable=E402
# pylint: enable=wildcard-import,wrong-import-position,unused-wildcard-import

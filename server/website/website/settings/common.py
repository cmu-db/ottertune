#
# OtterTune - common.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
"""
Common Django settings for the OtterTune project.

"""

import os
import sys
from datetime import timedelta
from os.path import abspath, dirname, exists, join

import djcelery

# ==============================================
# PATH CONFIGURATION
# ==============================================

# Absolute path to this Django project directory.
PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))

# Where the log files are stored
LOG_DIR = join(PROJECT_ROOT, 'log')

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
MEDIA_ROOT = join(PROJECT_ROOT, 'media')
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
STATIC_ROOT = join(PROJECT_ROOT, 'static')

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
            join(PROJECT_ROOT, 'website', 'templates')
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
    'django_db_logger',
)

# ==============================================
# DATABASE CONFIGURATION
# ==============================================

# Enables compression on MySQL >= 5.6 on tables:
#  - website_backupdata
#  - website_knobdata
#  - website_metricdata
#  - website_pipelinedata
MYSQL_COMPRESSION = False


# ==============================================
# RABBITMQ/CELERY CONFIGURATION
# ==============================================

# Broker URL for RabbitMq
BROKER_URL = 'amqp://guest:guest@localhost:5672//'

# Enable finer-grained reporting: will report 'started' when
# task is executed by a worker.
CELERY_TRACK_STARTED = True

# Do not let celery take over the root logger
CELERYD_HIJACK_ROOT_LOGGER = False

# Store celery results in the database
CELERY_RESULT_BACKEND = 'djcelery.backends.database:DatabaseBackend'

# The celerybeat scheduler class
CELERYBEAT_SCHEDULER = 'djcelery.schedulers.DatabaseScheduler'

# Defines the periodic task schedule for celerybeat
CELERYBEAT_SCHEDULE = {
    'run-every-5m': {
        'task': 'run_background_tasks',
        'schedule': timedelta(minutes=5),
    }
}

# The Celery documentation recommends disabling the rate limits
# if no tasks are using them
CELERY_DISABLE_RATE_LIMITS = True

# Worker will execute at most this many tasks before it's killed
# and replaced with a new worker. This helps with memory leaks.
CELERYD_MAX_TASKS_PER_CHILD = 20

# Number of concurrent workers. Defaults to the number of CPUs.
# CELERYD_CONCURRENCY = 8

# Late ack means the task messages will be acknowledged after
# the task has been executed, not just before
CELERY_ACKS_LATE = False

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
            'maxBytes': 2097152,
            'backupCount': 5,
            'formatter': 'standard',
        },
        'celery': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': join(LOG_DIR, 'celery.log'),
            'maxBytes': 2097152,
            'backupCount': 15,
            'formatter': 'standard',
        },
        'dblog': {
            'level': 'DEBUG',
            'class': 'django_db_logger.db_log_handler.DatabaseLogHandler',
        },
        'dblog_warn': {
            'level': 'WARN',
            'class': 'django_db_logger.db_log_handler.DatabaseLogHandler',
        },
        'console': {
            'level': 'DEBUG',
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
            'handlers': ['console', 'logfile', 'dblog_warn'],
            'propagate': True,
            'level': 'WARN',
        },
        'django.db.backends': {
            'handlers': ['console', 'logfile', 'dblog_warn'],
            'level': 'WARN',
            'propagate': False,
        },
        'website': {
            'handlers': ['console', 'logfile', 'dblog'],
            'level': 'DEBUG',
        },
        'django.request': {
            'handlers': ['console', 'dblog_warn'],
            'level': 'INFO',
            'propagate': False,
        },
        'celery': {
            'handlers': ['celery', 'dblog'],
            'level': 'DEBUG',
            'propogate': True,
        },
        'celery.task': {
            'handlers': ['celery', 'dblog'],
            'level': 'DEBUG',
            'propogate': True,
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

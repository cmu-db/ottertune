import json
import random
import string
from os import environ as env

debug = env.get('DEBUG', 'true').lower() == 'true'
rabbitmq_host = env.get('RABBITMQ_HOST', 'localhost')
backend = env.get('BACKEND', 'postgresql')
db_name = env.get('DB_NAME', 'ottertune')
db_host = env.get('DB_HOST', 'localhost')
db_pwd = env.get('DB_PASSWORD', '')

if backend == 'mysql':
    default_user = 'root'
    default_port = '3306'
    default_opts = {
        'init_command': "SET sql_mode='STRICT_TRANS_TABLES',innodb_strict_mode=1",
    }
else:
    default_user = 'postgres'
    default_port = '5432'
    default_opts = {}

db_user = env.get('DB_USER', default_user)
db_port = env.get('DB_PORT', default_port)
db_opts = env.get('DB_OPTS', default_opts)
if isinstance(db_opts, str):
    db_opts = json.loads(db_opts) if db_opts else {}

SECRET_KEY = ''.join(random.choice(string.hexdigits) for _ in range(16))
DATABASES = {
             'default': {'ENGINE': 'django.db.backends.' + backend,
                         'NAME': db_name,
                         'USER': db_user,
                         'PASSWORD': db_pwd,
                         'HOST': db_host,
                         'PORT': db_port,
                         'OPTIONS': db_opts,
                         }
             }
DEBUG = debug
ADMINS = ()
MANAGERS = ADMINS
ALLOWED_HOSTS = ['*']
BROKER_URL = 'amqp://guest:guest@{}:5672//'.format(rabbitmq_host)

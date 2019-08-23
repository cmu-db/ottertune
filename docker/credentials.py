import secrets
from os import environ as env

db_user = env.get('MYSQL_USER')
db_pwd = env.get('MYSQL_PASSWORD')
db_host = env.get('MYSQL_HOST')
db_port = env.get('MYSQL_PORT', '3306')
debug = env.get('DEBUG')

SECRET_KEY = secrets.token_hex(16)
DATABASES = {
             'default': {'ENGINE': 'django.db.backends.mysql',
                         'NAME': 'ottertune',
                         'USER': db_user,
                         'PASSWORD': db_pwd,
                         'HOST': db_host,
                         'PORT': db_port,
                         'OPTIONS': {'init_command': 'SET sql_mode=\'STRICT_TRANS_TABLES\',innodb_strict_mode=1',}
                         }
             }
DEBUG = True
ADMINS = ()
MANAGERS = ADMINS
ALLOWED_HOSTS = []

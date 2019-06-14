FROM ubuntu:18.04
  
ENV DEBIAN_FRONTEND=noninteractive
ENV DJANGO_SETTINGS_MODULE=website.settings
ENV C_FORCE_ROOT=true

EXPOSE 8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y git python3.6 python3-pip python3-tk mysql-server libmysqlclient-dev python-mysqldb rabbitmq-server parallel \
    && git clone https://github.com/cmu-db/ottertune.git

WORKDIR /app/ottertune/server/website

RUN touch website/settings/credentials.py \
    && echo "import secrets\nSECRET_KEY = secrets.token_hex(16)\nDATABASES = {'default': {'ENGINE': 'django.db.backends.mysql','NAME':'ottertune','USER':'root','PASSWORD':'','HOST':'','PORT':'','OPTIONS':{'init_command':'SET sql_mode=\'STRICT_TRANS_TABLES\',innodb_strict_mode=1',},}}\nDEBUG = True\nADMINS = ()\nMANAGERS = ADMINS\nALLOWED_HOSTS = []" >> website/settings/credentials.py \
    && pip3 install -r requirements.txt \
    && service mysql start \
    && mysqladmin create ottertune \
    && python3 manage.py makemigrations website \
    && python3 manage.py migrate \
    && python3 -c "import django; django.setup(); \
    from django.contrib.auth.management.commands.createsuperuser import get_user_model; \
    get_user_model()._default_manager.db_manager('default').create_superuser(username='root', email='email@domain.com', password='password')"

CMD service mysql start \
    && rabbitmq-server -detached \
    && parallel --lb -j 3 ::: 'python3 manage.py celery worker --loglevel=info --pool=threads' \
    'python3 manage.py celerybeat --verbosity=2 --loglevel=info' \
    'python3 manage.py runserver 0.0.0.0:8000'
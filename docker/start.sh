#!/bin/bash

# Wait for MySQL connection
/bin/bash wait-for-it.sh

## Needs a connection to a DB so migrations go here
python3 manage.py makemigrations website
python3 manage.py migrate
python3 createadmin.py

python3 manage.py celery worker --loglevel=info --pool=threads &
python3 manage.py celerybeat --verbosity=2 --loglevel=info &
python3 manage.py runserver 0.0.0.0:8000


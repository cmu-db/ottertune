#!/bin/bash

# Wait for MySQL connection
/bin/bash wait-for-it.sh

## Needs a connection to a DB so migrations go here
python3 manage.py makemigrations website
python3 manage.py migrate
python3 manage.py createuser admin $ADMIN_PASSWORD --superuser

python3 manage.py startcelery
python3 manage.py runserver 0.0.0.0:8000

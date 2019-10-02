python3 manage.py celery worker --loglevel=info --pool=threads --concurrency=1 > celery.log 2>&1 &

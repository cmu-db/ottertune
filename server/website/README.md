Website
=======

OLTP-Bench Website is an intermediate between the client's database and OtterTune (DBMS Auto-tuning system). 

## Requirements

##### Ubuntu Packages

```
sudo apt-get install python-pip python-dev python-mysqldb rabbitmq-server
```

##### Python Packages

```
sudo pip install -r requirements.txt
```

## Installation Instructions


##### 1. Update the Django settings

Navigate to the settings directory:

```
cd website/settings
```

Copy the credentials template:

```
cp credentials_TEMPLATE.py credentials.py
```

Edit `credentials.py` and update the secret key and database information.

##### 2. Serve the static files

If you do not use the website for production, simply set `DEBUG = True` in `credentials.py`. Then Django will handle static files automatically. 

This is not an efficient way for production. You need to configure other servers like Apache to serve static files in the production environment. ([Details](https://docs.djangoproject.com/en/1.11/howto/static-files/deployment/))

##### 3. Create the MySQL database if it does not already exist

```
mysqladmin create -u <username> -p ottertune
```

##### 4. Migrate the Django models into the database

```
python manage.py makemigrations website
python manage.py migrate
```

##### 5. Create the super user

```
python manage.py createsuperuser
```
    
##### 6. Start the message broker, celery worker, website server, and periodic task

```
sudo rabbitmq-server -detached
python manage.py celery worker --loglevel=info --pool=threads
python manage.py runserver 0.0.0.0:8000
python manage.py celerybeat --verbosity=2 --loglevel=info 

```

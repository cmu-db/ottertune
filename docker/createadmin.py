#!/usr/bin/env python
  
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "website.settings")
django.setup()

from django.contrib.auth.models import User

username = os.environ.get('ADMIN_USER', 'admin')
password = os.environ.get('ADMIN_PASSWORD')
email = os.environ.get('ADMIN_EMAIL', 'admin@example.com')

if password:
  if not User.objects.filter(username=username).exists():
     print(f"Creating '{username}' user...")
     User.objects.create_superuser(username=username,
                                   password=password,
                                   email=email)
     print(f"'{username}' user created!")
  else:
     print(f"'{username}' user already exists! Setting '{username}' password")
     u = User.objects.get(username=username)
     u.set_password(password)
     u.save()

#!/bin/bash

# Move to Dockerfile
pip3 install coverage codecov

HOME_DIR=$(pwd)
GRADLE_VERSION=gradle-5.5.1
GRADLE_HOME=/opt/${GRADLE_VERSION}

wget https://services.gradle.org/distributions/${GRADLE_VERSION}-bin.zip
unzip ${GRADLE_VERSION}-bin.zip -d /opt
$GRADLE_HOME/bin/gradle --version

# Wait for MySQL connection
/bin/bash wait-for-it.sh

cd ../../client/controller && $GRADLE_HOME/bin/gradle build
cd $HOME_DIR
cd ../../server && python3 -m unittest discover -s analysis/tests -v
cd $HOME_DIR
coverage run manage.py test --noinput -v 2
cd ../.. && python3 script/validators/source_validator.py

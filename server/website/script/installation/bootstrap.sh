#!/usr/bin/env bash

# Variables
DBHOST=localhost
DBNAME=ottertune
DBUSER=dbuser
DBPASSWD=test123

LOG=/vagrant/vm_build.log
REPOPATH=/ottertune
SETTINGSPATH=$REPOPATH/server/website/website/settings

# Clear old log contents
> $LOG

# Install Ubuntu packages
echo -e "\n--- Installing Ubuntu packages ---\n"
apt-get -qq update
apt-get -y install python3-pip python-dev python-mysqldb rabbitmq-server gradle default-jdk libmysqlclient-dev python3-tk >> $LOG 2>&1

echo -e "\n--- Installing Python packages ---\n"
pip3 install --upgrade pip >> $LOG 2>&1
pip install -r ${REPOPATH}/server/website/requirements.txt >> $LOG 2>&1

# Install MySQL
echo -e "\n--- Install MySQL specific packages and settings ---\n"
debconf-set-selections <<< "mysql-server mysql-server/root_password password $DBPASSWD"
debconf-set-selections <<< "mysql-server mysql-server/root_password_again password $DBPASSWD"
apt-get -y install mysql-server >> $LOG 2>&1

# Setup MySQL
echo -e "\n--- Setting up the MySQL user and database ---\n"
mysql -uroot -p$DBPASSWD -e "CREATE DATABASE IF NOT EXISTS $DBNAME" >> /vagrant/vm_build.log 2>&1
mysql -uroot -p$DBPASSWD -e "GRANT ALL PRIVILEGES ON $DBNAME.* TO '$DBUSER'@'localhost' IDENTIFIED BY '$DBPASSWD'" >> $LOG 2>&1
mysql -uroot -p$DBPASSWD -e "GRANT ALL PRIVILEGES ON test_$DBNAME.* TO '$DBUSER'@'localhost' IDENTIFIED BY '$DBPASSWD'" >> $LOG 2>&1

# Update Django settings
echo -e "\n--- Updating Django settings ---\n"
if [ ! -f "$SETTINGSPATH/credentials.py" ]; then
    cp $SETTINGSPATH/credentials_TEMPLATE.py $SETTINGSPATH/credentials.py >> $LOG 2>&1
    sed -i -e "s/^DEBUG.*/DEBUG = True/" \
        -e "s/^ALLOWED_HOSTS.*/ALLOWED_HOSTS = ['0\.0\.0\.0']/" \
        -e "s/'USER': 'ADD ME\!\!'/'USER': '$DBUSER'/" \
        -e "s/'PASSWORD': 'ADD ME\!\!'/'PASSWORD': '$DBPASSWD'/" \
        $SETTINGSPATH/credentials.py >> $LOG 2>&1
fi
rm /usr/bin/python
ln -s /usr/bin/python3.5 /usr/bin/python
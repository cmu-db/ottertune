#!/bin/sh
# wait until MySQL is really available
maxcounter=${MAX_DB_CONN_ATTEMPTS:-45}
echo "Trying to connect to mysql, max attempts="$maxcounter
 
counter=1
while ! mysql --host="$MYSQL_HOST" --protocol TCP -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" -e "show databases;" > /dev/null 2>&1; do
    sleep 1
    counter=`expr $counter + 1`
    if [ $counter -gt $maxcounter ]; then
        >&2 echo "We have been waiting for MySQL too long already; failing."
        exit 1
    fi;
done
echo "-=------------------------------------------------------"
echo "-=------------------------------------------------------"
echo "Connected to MySQL!"
echo "-=------------------------------------------------------"
echo "-=------------------------------------------------------"

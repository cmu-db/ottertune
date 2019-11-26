#!/bin/sh

maxcounter=${MAX_DB_CONN_ATTEMPTS:-60}

if [ "$maxcounter" -le 0 ]; then
    echo "Skipping wait-for-it.sh..."
    exit 0
fi

if [ -z "$BACKEND" ]; then
    echo "ERROR: variable 'BACKEND' must be set. Exiting."
    exit 1
fi

# wait until the database is really available
echo "Trying to connect to $BACKEND (timeout=${maxcounter}s)"
echo ""

ready () {

    if [ "$BACKEND" = "mysql" ]; then
        mysql \
            --host="$DB_HOST" \
            --protocol TCP \
            -u"$DB_USER" \
            -p"$DB_PASSWORD" \
            -e "show databases;" > /dev/null 2>&1
    else
        PGPASSWORD="$DB_PASSWORD" psql \
            -h "$DB_HOST" \
            -U "$DB_USER" \
            -c "select * from pg_database" > /dev/null 2>&1
    fi
    return $?
}

 
counter=1
while ! ready; do
    counter=`expr $counter + 1`

    if [ $counter -gt $maxcounter ]; then
        >&2 echo "ERROR: Could not connect to $BACKEND after $MAX_DB_CONN_ATTEMPTS seconds; Exiting."
        exit 1
    fi;
    sleep 1
done

echo "-=------------------------------------------------------"
echo "-=------------------------------------------------------"
echo "Connected to $BACKEND!"
echo "-=------------------------------------------------------"
echo "-=------------------------------------------------------"

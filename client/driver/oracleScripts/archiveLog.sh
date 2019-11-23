#!/bin/sh

# Change log mode
sqlplus / as sysdba <<EOF
    shutdown immediate
    startup mount
    ALTER DATABASE ARCHIVELOG;
    ALTER DATABASE OPEN;
quit
EOF


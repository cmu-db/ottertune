#!/bin/sh
su - oracle <<EON
oracle
sqlplus / as sysdba <<EOF
@/home/oracle/ottertune/client/driver/autoawr.sql;
quit
EOF
exit  
EON

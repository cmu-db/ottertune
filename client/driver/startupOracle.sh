#!/bin/sh
su - oracle <<EON
sqlplus / as sysdba <<EOF
startup
quit
EOF
exit  
EON

#!/bin/sh
su - oracle <<EON
sqlplus / as sysdba <<EOF
shutdown immediate
exit
EOF
exit  
EON

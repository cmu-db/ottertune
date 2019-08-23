#!/bin/sh
su - oracle <<EON
oracle

sqlplus / as sysdba <<EOF
startup
quit
EOF

exit  
EON

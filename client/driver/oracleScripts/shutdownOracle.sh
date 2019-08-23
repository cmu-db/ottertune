#!/bin/sh
su - oracle <<EON
oracle

sqlplus / as sysdba <<EOF
shutdown immediate
exit
EOF

exit  
EON

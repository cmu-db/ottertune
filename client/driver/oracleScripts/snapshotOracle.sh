#!/bin/sh
su - oracle <<EON
oracle

sqlplus / as sysdba <<EOF
exec dbms_workload_repository.create_snapshot;
quit
EOF

exit  
EON

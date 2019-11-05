#!/bin/sh

sqlplus / as sysdba <<EOF
exec dbms_workload_repository.create_snapshot;
quit
EOF


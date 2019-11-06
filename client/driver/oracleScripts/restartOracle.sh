#!/bin/sh

sqlplus / as sysdba <<EOF
shutdown immediate
startup
quit
EOF


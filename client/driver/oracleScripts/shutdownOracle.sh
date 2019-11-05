#!/bin/sh

sqlplus / as sysdba <<EOF
shutdown immediate
exit
EOF


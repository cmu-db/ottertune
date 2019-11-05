#!/bin/sh

USERNAME="$1"

sqlplus / as sysdba <<EOF
drop user $USERNAME cascade;	
quit
EOF


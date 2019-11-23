#!/bin/sh

RESTORE_POINT="$1"

# Remove RESTORE Point
sqlplus / as sysdba <<EOF
    'DROP RESTORE POINT $RESTORE_POINT
quit
EOF


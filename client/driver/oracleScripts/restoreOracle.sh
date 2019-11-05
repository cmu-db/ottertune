#!/bin/sh

USERNAME="$1"
DP_FILE="${2}.dump"
DP_DIR="DPDATA"

# Import the data
impdp 'userid="/ as sysdba"' \
    schemas=$USERNAME \
    dumpfile=$DP_FILE \
    DIRECTORY=$DP_DIR

# Restart the database
sqlplus / as sysdba <<EOF
shutdown immediate
startup
quit
EOF


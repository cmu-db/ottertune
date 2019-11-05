#!/bin/sh

USERNAME="$1"
PASSWORD="$2"
DB_NAME="$3"
DP_PATH="$4"
DP_DIR="DPDATA"
DP_FILE="${DB_NAME}.dump"

# Make sure the physical directory exists
mkdir -p "$DP_PATH"

# Make sure the DB directory object exists
sqlplus / as sysdba <<EOF
CREATE OR REPLACE DIRECTORY $DP_DIR AS '$DP_PATH';
quit
EOF

# Export the data
expdp $USERNAME/$PASSWORD@$DB_NAME \
    schemas=$USERNAME \
    dumpfile=$DP_FILE \
    DIRECTORY=$DP_DIR


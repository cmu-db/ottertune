#!/bin/sh
LOGFILE="$1"

sqlplus / as sysdba <<EOF
shutdown immediate
spool $LOGFILE
startup
spool off
quit
EOF


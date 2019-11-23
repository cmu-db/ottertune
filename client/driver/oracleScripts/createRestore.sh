#!/bin/sh

RESTORE_POINT="$1"
SIZE="$2"
RECOVERY_FILE="$3"

# Make sure the physical directory exists
mkdir -p "$DP_PATH"

# Set recovery file and restore point
sqlplus / as sysdba <<EOF
    DECLARE
      rfdexists INTEGER;
    BEGIN
      SELECT COUNT(*) INTO rfdexists FROM V\$RECOVERY_FILE_DEST WHERE name='$RECOVERY_FILE';
      IF (rfdexists = 0) then
        EXECUTE IMMEDIATE 'ALTER SYSTEM SET DB_RECOVERY_FILE_DEST_SIZE = $SIZE';
        EXECUTE IMMEDIATE 'ALTER SYSTEM SET DB_RECOVERY_FILE_DEST = $RECOVERY_FILE';
      END IF;
      DBMS_OUTPUT.PUT_LINE(rfdexists);
    END;
    DECLARE
      rpexists INTEGER;
    BEGIN
      SELECT COUNT(*) INTO rpexists FROM v\\\\\$restore_point WHERE name='$RESTORE_POINT';
      IF (rpexists = 0) then
        EXECUTE IMMEDIATE 'CREATE RESTORE POINT $RESTORE_POINT GUARANTEE FLASHBACK DATABASE';
      END IF;
      DBMS_OUTPUT.PUT_LINE(rpexists);
    END;
quit
EOF


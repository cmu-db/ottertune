set colsep ,
set headsep off
set pagesize 0
set trimspool on
set linesize 4100
set numwidth 19
set heading on
set feed off
set echo off
set tab off
set trim on
set trims on

-- This option is only available in versions 12.2 and up
set markup csv on

spool knob_info.csv

select * from v$parameter order by name;

spool off


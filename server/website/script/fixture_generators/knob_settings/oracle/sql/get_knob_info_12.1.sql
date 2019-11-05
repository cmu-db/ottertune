SET echo off
SET linesize 32767
SET LONG 90000
SET LONGCHUNKSIZE 90000;
SET wrap off;
SET heading off  
SET colsep '|' 
SET pagesize 0;
SET feed off;
SET termout off;
SET trimspool ON;
SELECT * FROM v$parameter order by name;
spool t2.csv
/
spool off

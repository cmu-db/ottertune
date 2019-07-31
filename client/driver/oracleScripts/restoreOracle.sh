#!/bin/sh
su - oracle <<EON
oracle	#system password

sqlplus / as sysdba <<EOF
drop user c##tpcc cascade;	
#         username
create user c##tpcc identified by oracle;	
#          username              password
quit
EOF

impdp 'userid="/ as sysdba"' schemas=c##tpcc dumpfile=orcldb.dump DIRECTORY=dpdata
#                                    username        database_name       db_directory

sqlplus / as sysdba <<EOF	#restart the database
shutdown immediate
startup
quit
EOF

exit  
EON

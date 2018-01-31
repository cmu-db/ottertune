package com.controller;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

/**
 * Database Type.
 */
public enum DatabaseType {

    /**
     * Parameters:
     * (1) JDBC Driver String
     */
    DB2("com.ibm.db2.jcc.DB2Driver"),
    MYSQL("com.mysql.jdbc.Driver"),
    MYROCKS("com.mysql.jdbc.Driver"),
    POSTGRES("org.postgresql.Driver"),
    ORACLE("oracle.jdbc.driver.OracleDriver"),
    SQLSERVER("com.microsoft.sqlserver.jdbc.SQLServerDriver"),
    SQLITE("org.sqlite.JDBC"),
    AMAZONRDS(null),
    SQLAZURE(null),
    ASSCLOWN(null),
    HSQLDB("org.hsqldb.jdbcDriver"),
    H2("org.h2.Driver"),
    MONETDB("nl.cwi.monetdb.jdbc.MonetDriver"),
    NUODB("com.nuodb.jdbc.Driver"),
    TIMESTEN("com.timesten.jdbc.TimesTenDriver"),
    PELOTON("org.postgresql.Driver")
    ;

    private DatabaseType(String driver) {
        this.driver = driver;
    }

    /**
     * This is the suggested driver string to use in the configuration xml
     * This corresponds to the <B>'driver'</b> attribute.
     */
    private final String driver;


    // ---------------------------------------------------------------
    // ACCESSORS
    // ----------------------------------------------------------------

    /**
     * Returns the suggested driver string to use for the given database type
     * @return
     */
    public String getSuggestedDriver() {
        return (this.driver);
    }

    // ----------------------------------------------------------------
    // STATIC METHODS + MEMBERS
    // ----------------------------------------------------------------

    protected static final Map<Integer, DatabaseType> idx_lookup = new HashMap<Integer, DatabaseType>();
    protected static final Map<String, DatabaseType> name_lookup = new HashMap<String, DatabaseType>();
    static {
        for (DatabaseType vt : EnumSet.allOf(DatabaseType.class)) {
            DatabaseType.idx_lookup.put(vt.ordinal(), vt);
            DatabaseType.name_lookup.put(vt.name().toUpperCase(), vt);
        }
    }

    public static DatabaseType get(String name) {
        DatabaseType ret = DatabaseType.name_lookup.get(name.toUpperCase());
        return (ret);
    }

}

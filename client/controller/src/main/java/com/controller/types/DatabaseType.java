/*
 * OtterTune - DatabaseType.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

/** Database Type. */
public enum DatabaseType {

  /** Parameters: (1) JDBC Driver String */
  MYSQL("com.mysql.jdbc.Driver"),
  MYROCKS("com.mysql.jdbc.Driver"),
  POSTGRES("org.postgresql.Driver"),
  SAPHANA("com.sap.db.jdbc.Driver"),
  ORACLE("oracle.jdbc.driver.OracleDriver");

  private DatabaseType(String driver) {
    this.driver = driver;
  }

  /**
   * This is the suggested driver string to use in the configuration xml This corresponds to the
   * <B>'driver'</b> attribute.
   */
  private final String driver;

  // ---------------------------------------------------------------
  // ACCESSORS
  // ----------------------------------------------------------------

  /**
   * Returns the suggested driver string to use for the given database type
   *
   * @return
   */
  public String getSuggestedDriver() {
    return (this.driver);
  }

  // ----------------------------------------------------------------
  // STATIC METHODS + MEMBERS
  // ----------------------------------------------------------------

  protected static final Map<Integer, DatabaseType> idx_lookup =
      new HashMap<Integer, DatabaseType>();
  protected static final Map<String, DatabaseType> name_lookup =
      new HashMap<String, DatabaseType>();

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

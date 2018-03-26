/*
 * OtterTune - ControllerConfiguration.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller;

/** Controller Configuration. */
public class ControllerConfiguration {
  private DatabaseType dbType;
  private String dbName;
  private String dbUsername;
  private String dbPassword;
  private String dbURL;
  private String uploadCode;
  private String uploadURL;
  private String workloadName;

  public ControllerConfiguration() {}

  public ControllerConfiguration(
      String dbName,
      String dbUsername,
      String dbPassword,
      String dbURL,
      String uploadCode,
      String uploadURL,
      String workloadName) {
    this.dbType = DatabaseType.get(dbName);
    this.dbName = dbName;
    this.dbUsername = dbUsername;
    this.dbPassword = dbPassword;
    this.dbURL = dbURL;
    this.uploadCode = uploadCode;
    this.uploadURL = uploadURL;
    this.workloadName = workloadName;
  }

  /* Mutators */
  public void setDBType(DatabaseType dbType) {
    this.dbType = dbType;
  }

  public void setDBName(String dbName) {
    this.dbName = dbName;
  }

  public void setDBUsername(String dbUsername) {
    this.dbUsername = dbUsername;
  }

  public void setPassword(String dbPassword) {
    this.dbPassword = dbPassword;
  }

  public void setDBURL(String dbURL) {
    this.dbURL = dbURL;
  }

  public void setUploadCode(String uploadCode) {
    this.uploadCode = uploadCode;
  }

  public void setUploadURL(String uploadURL) {
    this.uploadURL = uploadURL;
  }

  public void setWorkloadName(String workloadName) {
    this.workloadName = workloadName;
  }

  /* Getters */
  public DatabaseType getDBType() {
    return this.dbType;
  }

  public String getDBName() {
    return this.dbName;
  }

  public String getDBUsername() {
    return this.dbUsername;
  }

  public String getDBPassword() {
    return this.dbPassword;
  }

  public String getDBURL() {
    return this.dbURL;
  }

  public String getUploadCode() {
    return this.uploadCode;
  }

  public String getUploadURL() {
    return this.uploadURL;
  }

  public String getWorkloadName() {
    return this.workloadName;
  }
}

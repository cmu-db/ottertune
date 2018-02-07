/*
 * OtterTune - ControllerConfiguration.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller;

/** Controller Configuration. */
public class ControllerConfiguration {
  private DatabaseType dbType;
  private String dbUsername;
  private String dbPassword;
  private String dbDriver;
  private String uploadCode;
  private String uploadURL;
  private String workloadName;

  public ControllerConfiguration() {}

  public ControllerConfiguration(
      DatabaseType dbType,
      String dbUsername,
      String dbPassword,
      String dbDriver,
      String uploadCode,
      String uploadURL,
      String workloadName) {
    this.dbType = dbType;
    this.dbUsername = dbUsername;
    this.dbPassword = dbPassword;
    this.dbDriver = dbDriver;
    this.uploadCode = uploadCode;
    this.uploadURL = uploadURL;
    this.workloadName = workloadName;
  }
  /* Mutators */
  public void setDBType(DatabaseType dbType) {
    this.dbType = dbType;
  }

  public void setDBUsername(String dbUsername) {
    this.dbUsername = dbUsername;
  }

  public void setPassword(String dbPassword) {
    this.dbPassword = dbPassword;
  }

  public void setDBDriver(String dbDriver) {
    this.dbDriver = dbDriver;
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

  public String getDBUsername() {
    return this.dbUsername;
  }

  public String getDBPassword() {
    return this.dbPassword;
  }

  public String getDBDriver() {
    return this.dbDriver;
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

package com.controller;

/**
 * Controller Configuration.
 */
public class ControllerConfiguration {
    private DatabaseType dbtype;
    private String username;
    private String password;
    private String databaseUrl;
    private String uploadCode;
    private String uploadURL;
    private String workloadName;


    public ControllerConfiguration() {}

    public ControllerConfiguration(DatabaseType dbtype, String username,
                                   String password, String databaseUrl,
                                   String uploadCode, String uploadURL,
                                   String workloadName) {
        this.dbtype = dbtype;
        this.username = username;
        this.password = password;
        this.databaseUrl = databaseUrl;
        this.uploadCode = uploadCode;
        this.uploadURL = uploadURL;
        this.workloadName = workloadName;
    }
    /* Mutators */
    public void setDbtype(DatabaseType dbtype) {
        this.dbtype = dbtype;
    }
    public void setUsername(String username) {
        this.username = username;
    }
    public void setPassword(String password) {
        this.password = password;
    }
    public void setDatabaseUrl(String dburl) {
        this.databaseUrl = dburl;
    }
    public void setUploadCode(String upcode) {
        this.uploadCode = upcode;
    }
    public void setUploadURL(String upurl) {
        this.uploadURL = upurl;
    }
    public void setWorkloadName(String workloadName) {
        this.workloadName = workloadName;
    }

    /* Getters */
    public DatabaseType getDbtype() {
        return this.dbtype;
    }
    public String getUsername() {
        return this.username;
    }
    public String getPassword() {
        return this.password;
    }
    public String getDatabaseUrl() {
        return this.databaseUrl;
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

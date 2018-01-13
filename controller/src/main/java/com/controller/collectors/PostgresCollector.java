/******************************************************************************
 *  Copyright 2015 by OLTPBenchmark Project                                   *
 *                                                                            *
 *  Licensed under the Apache License, Version 2.0 (the "License");           *
 *  you may not use this file except in compliance with the License.          *
 *  You may obtain a copy of the License at                                   *
 *                                                                            *
 *    http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                            *
 *  Unless required by applicable law or agreed to in writing, software       *
 *  distributed under the License is distributed on an "AS IS" BASIS,         *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 *  See the License for the specific language governing permissions and       *
 *  limitations under the License.                                            *
 ******************************************************************************/

package com.controller.collectors;

import com.controller.util.JSONUtil;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

import java.io.IOException;
import java.sql.*;
import java.util.*;

public class PostgresCollector extends DBCollector {
    private static final Logger LOG = LogManager.getLogger(PostgresCollector.class);

    private static final String VERSION_SQL = "SELECT version();";

    private static final String PARAMETERS_SQL = "SHOW ALL;";

    private static final String[] PG_STAT_VIEWS = {
    	    "pg_stat_archiver",
            "pg_stat_bgwriter",
            "pg_stat_database",
    	    "pg_stat_database_conflicts",
            "pg_stat_user_tables",
            "pg_statio_user_tables",
            "pg_stat_user_indexes",
            "pg_statio_user_indexes"
    };

    private static final String[] PG_STAT_VIEWS_LOCAL_DATABASE = {"pg_stat_database", "pg_stat_database_conflicts"};
    private static final String PG_STAT_VIEWS_LOCAL_DATABASE_KEY = "datname";
    private static final String[] PG_STAT_VIEWS_LOCAL_TABLE = {"pg_stat_user_tables", "pg_statio_user_tables"};
    private static final String PG_STAT_VIEWS_LOCAL_TABLE_KEY = "relname";
    private static final String[] PG_STAT_VIEWS_LOCAL_INDEXES = {"pg_stat_user_indexes", "pg_statio_user_indexes"};
    private static final String PG_STAT_VIEWS_LOCAL_INDEXES_KEY = "relname";

    private final Map<String, List<Map<String, String>>> pgMetrics;

    public PostgresCollector(String oriDBUrl, String username, String password) {
    	pgMetrics = new HashMap<>();
        try {
            Connection conn = DriverManager.getConnection(oriDBUrl, username, password);
//            Catalog.setSeparator(conn);

            Statement s = conn.createStatement();

            // Collect DBMS version
            ResultSet out = s.executeQuery(VERSION_SQL);
            if (out.next()) {
            	this.version.append(out.getString(1));
            }

            // Collect DBMS parameters
            out = s.executeQuery(PARAMETERS_SQL);
            while (out.next()) {
                dbParameters.put(out.getString("name"), out.getString("setting"));
            }

            // Collect DBMS internal metrics
            for (String viewName : PG_STAT_VIEWS) {
            	out = s.executeQuery("SELECT * FROM " + viewName);
            	pgMetrics.put(viewName, getMetrics(out));
            }
        } catch (SQLException e) {
            LOG.error("Error while collecting DB parameters: " + e.getMessage());
        }
    }

    @Override
    public boolean hasMetrics() {
    	return !pgMetrics.isEmpty();
    }

    private Map<String, Map<String, String>> getMetricsMap(String viewName, String jsonKeyName) {
        Map<String, Map<String, String>> thisViewObj = new HashMap<>();

        List<Map<String, String>> thisViewList = pgMetrics.get(viewName);
        for(Map<String, String> dbmap : thisViewList) {
            String jsonkey = dbmap.get(jsonKeyName);
            thisViewObj.put(jsonkey, dbmap);
        }
        return thisViewObj;
    }

    @Override
    public String collectMetrics() {

        // create global objects for two views: "pg_stat_archiver" and "pg_stat_bgwriter"
        Map<String, Map<String, String>> jobGlobal = new HashMap<>();

        // "pg_stat_archiver" (only one instance in the list)
        Map<String, String> archiverList = pgMetrics.get("pg_stat_archiver").get(0);
        jobGlobal.put("pg_stat_archiver", archiverList);

        // "pg_stat_bgwriter" (only one instance in the list)
        Map<String, String> bgwriterList = pgMetrics.get("pg_stat_bgwriter").get(0);
        jobGlobal.put("pg_stat_bgwriter", bgwriterList);

        Map<String, Object> jobLocal = new HashMap<>();
        jobLocal.put(JSON_GLOBAL_KEY, jobGlobal);

        // "table"
        Map<String, Object> jobTable = new HashMap<>();
        for (String viewName : PG_STAT_VIEWS_LOCAL_TABLE) {
            String jsonKeyName = PG_STAT_VIEWS_LOCAL_TABLE_KEY;
            jobTable.put(viewName, getMetricsMap(viewName, jsonKeyName));
        }

        jobLocal.put("table", jobTable);

        Map<String, Object> jobDatabase = new HashMap<>();

        // "database"
        for (String viewName : PG_STAT_VIEWS_LOCAL_DATABASE) {
            String jsonKeyName = PG_STAT_VIEWS_LOCAL_DATABASE_KEY;
            jobDatabase.put(viewName, getMetricsMap(viewName, jsonKeyName));
        }
        jobLocal.put("database", jobDatabase);

        // "indexes"
        Map<String, Object> jobIndexes = new HashMap<>();
        for (String viewName : PG_STAT_VIEWS_LOCAL_INDEXES) {
            String jsonKeyName = PG_STAT_VIEWS_LOCAL_INDEXES_KEY;
            jobIndexes.put(viewName, getMetricsMap(viewName, jsonKeyName));
        }
        jobLocal.put("indexes",jobIndexes);

        try {
            return JSONUtil.format(jobLocal);
        }
        catch (JsonProcessingException e) {
            e.printStackTrace();
        }
        return null;
    }

    private static List<Map<String, String>> getMetrics(ResultSet out) throws SQLException {
        ResultSetMetaData metadata = out.getMetaData();
        int numColumns = metadata.getColumnCount();
        String[] columnNames = new String[numColumns];
        for (int i = 0; i < numColumns; ++i) {
        	columnNames[i] = metadata.getColumnName(i + 1).toLowerCase();
        }

        List<Map<String, String>> metrics = new ArrayList<>();
        while (out.next()) {
        	Map<String, String> metricMap = new TreeMap<>();
        	for (int i = 0; i < numColumns; ++i) {
        		metricMap.put(columnNames[i], out.getString(i + 1));
        	}
        	metrics.add(metricMap);
        }
        return metrics;
    }

    @Override
    public String collectParameters() {
        String json = null;
        try {
            final ObjectMapper mapper = new ObjectMapper();
            Map<String, String> job = new HashMap<>();

            String globalJson = mapper.writeValueAsString(dbParameters);

            job.put(JSON_GLOBAL_KEY, globalJson);
            job.put(JSON_LOCAL_KEY, null);
            json = mapper.writeValueAsString(job);
            return JSONUtil.format(json);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        return json;
    }


}

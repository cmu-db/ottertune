/*
 * OtterTune - SAPHanaCollector.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller.collectors;

import com.controller.util.JSONUtil;
import com.controller.util.json.JSONException;
import com.controller.util.json.JSONObject;
import com.controller.util.json.JSONStringer;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.apache.log4j.Logger;

public class SAPHanaCollector extends DBCollector {
  private static final Logger LOG = Logger.getLogger(SAPHanaCollector.class);

  private static final String VERSION_SQL = "SELECT VERSION from M_DATABASE";

  private static final String PARAMETERS_SQL = "Select * from M_INIFILE_CONTENTS";

  private static final String[] SAP_SYS_VIEW = {
    "m_host_agent_metrics",
    "m_caches",
    "m_disk_usage",
    "m_garbage_collection_statistics",
    "m_host_resource_utilization",
    "m_data_volumes"
  };

  private static final String[] SAP_SYS_LOCAL_VIEW = {"m_table_statistics", "m_rs_indexes"};
  private static final String[] SAP_SYS_VIEW_GLOBAL = {
    "m_host_agent_metrics",
    "m_caches",
    "m_disk_usage",
    "m_garbage_collection_statistics",
    "m_host_resource_utilization",
    "m_data_volumes"
  };
  private static final String[] SAP_SYS_VIEW_GLOBAL_KEY = {
    "instance_id", "cache_id", "usage_type", "store_type", "host", "volume_id"
  };
  private static final String[] SAP_SYS_VIEW_LOCAL_TABLE = {"m_table_statistics"};
  private static final String[] SAP_SYS_VIEW_LOCAL_TABLE_KEY = {"table_name"};
  private static final String[] SAP_SYS_VIEW_LOCAL_INDEXES = {"m_rs_indexes"};
  private static final String[] SAP_SYS_VIEW_LOCAL_INDEXES_KEY = {"index_name"};

  private final Map<String, List<Map<String, String>>> pgMetrics;

  public SAPHanaCollector(String oriDBUrl, String username, String password) {
    pgMetrics = new HashMap<>();
    try {
      Connection conn = DriverManager.getConnection(oriDBUrl, username, password);
      Statement s = conn.createStatement();

      // Collect DBMS version
      ResultSet out = s.executeQuery(VERSION_SQL);
      if (out.next()) {
        this.version.append(out.getString(1));
      }

      // Collect DBMS parameters
      out = s.executeQuery(PARAMETERS_SQL);
      while (out.next()) {
        dbParameters.put(
            "("
                + out.getString("FILE_NAME")
                + ","
                + out.getString("LAYER_NAME")
                + ","
                + out.getString("TENANT_NAME")
                + ","
                + out.getString("HOST")
                + ","
                + out.getString("SECTION")
                + ","
                + out.getString("KEY")
                + ")",
            out.getString("VALUE"));
      }

      // Collect DBMS internal metrics
      for (String viewName : SAP_SYS_VIEW) {
        out = s.executeQuery("SELECT * FROM " + viewName);
        pgMetrics.put(viewName, getMetrics(out));
      }
      for (String viewName : SAP_SYS_LOCAL_VIEW) {
        out = s.executeQuery("SELECT * FROM " + viewName + " where schema_name = 'SYSTEM' ");
        pgMetrics.put(viewName, getMetrics(out));
      }
      conn.close();
    } catch (SQLException e) {
      e.printStackTrace();
      LOG.error("Error while collecting DB parameters: " + e.getMessage());
    }
  }

  @Override
  public boolean hasMetrics() {
    return (pgMetrics.isEmpty() == false);
  }

  private JSONObject genMapJSONObj(Map<String, String> mapin) {
    JSONObject res = new JSONObject();
    try {
      for (String key : mapin.keySet()) {
        res.put(key, mapin.get(key));
      }
    } catch (JSONException je) {
      LOG.error(je);
    }
    return res;
  }

  private JSONObject genLocalJSONObj(String viewName, String jsonKeyName) {
    JSONObject thisViewObj = new JSONObject();
    List<Map<String, String>> thisViewList = pgMetrics.get(viewName);
    try {
      for (Map<String, String> dbmap : thisViewList) {
        String jsonkey = dbmap.get(jsonKeyName);
        thisViewObj.put(jsonkey, genMapJSONObj(dbmap));
      }
    } catch (JSONException je) {
      LOG.error(je);
    }
    return thisViewObj;
  }

  @Override
  public String collectMetrics() {
    JSONStringer stringer = new JSONStringer();
    try {
      stringer.object();
      stringer.key(JSON_GLOBAL_KEY);
      JSONObject jobGlobal = new JSONObject();

      JSONObject jobMetric = new JSONObject();
      for (int i = 0; i < SAP_SYS_VIEW_GLOBAL.length; i++) {
        String viewName = SAP_SYS_VIEW_GLOBAL[i];
        String jsonKeyName = SAP_SYS_VIEW_GLOBAL_KEY[i];
        jobGlobal.put(viewName, genLocalJSONObj(viewName, jsonKeyName));
      }
      // add global json object
      stringer.value(jobGlobal);
      stringer.key(JSON_LOCAL_KEY);

      // create local objects for the rest of the views
      JSONObject jobLocal = new JSONObject();

      // "table"
      JSONObject jobTable = new JSONObject();
      for (int i = 0; i < SAP_SYS_VIEW_LOCAL_TABLE.length; i++) {
        String viewName = SAP_SYS_VIEW_LOCAL_TABLE[i];
        String jsonKeyName = SAP_SYS_VIEW_LOCAL_TABLE_KEY[i];
        jobTable.put(viewName, genLocalJSONObj(viewName, jsonKeyName));
      }
      jobLocal.put("table", jobTable);

      // "indexes"
      JSONObject jobIndexes = new JSONObject();
      for (int i = 0; i < SAP_SYS_VIEW_LOCAL_INDEXES.length; i++) {
        String viewName = SAP_SYS_VIEW_LOCAL_INDEXES[i];
        String jsonKeyName = SAP_SYS_VIEW_LOCAL_INDEXES_KEY[i];
        jobIndexes.put(viewName, genLocalJSONObj(viewName, jsonKeyName));
      }
      jobLocal.put("indexes", jobIndexes);

      // add local json object
      stringer.value(jobLocal);
      stringer.endObject();

    } catch (JSONException jsonexn) {
      jsonexn.printStackTrace();
    }

    return JSONUtil.format(stringer.toString());
  }

  private static List<Map<String, String>> getMetrics(ResultSet out) throws SQLException {
    ResultSetMetaData metadata = out.getMetaData();
    int numColumns = metadata.getColumnCount();
    String[] columnNames = new String[numColumns];
    for (int i = 0; i < numColumns; ++i) {
      columnNames[i] = metadata.getColumnName(i + 1).toLowerCase();
    }

    List<Map<String, String>> metrics = new ArrayList<Map<String, String>>();
    while (out.next()) {
      Map<String, String> metricMap = new TreeMap<String, String>();
      for (int i = 0; i < numColumns; ++i) {
        metricMap.put(columnNames[i], out.getString(i + 1));
      }
      metrics.add(metricMap);
    }
    return metrics;
  }

  @Override
  public String collectParameters() {
    JSONStringer stringer = new JSONStringer();
    try {
      stringer.object();
      stringer.key(JSON_GLOBAL_KEY);
      JSONObject jobLocal = new JSONObject();
      JSONObject job = new JSONObject();
      for (String k : dbParameters.keySet()) {
        job.put(k, dbParameters.get(k));
      }
      // "global is a fake view_name (a placeholder)"
      jobLocal.put("global", job);
      stringer.value(jobLocal);
      stringer.key(JSON_LOCAL_KEY);
      stringer.value(null);
      stringer.endObject();
    } catch (JSONException jsonexn) {
      jsonexn.printStackTrace();
    }
    return JSONUtil.format(stringer.toString());
  }
}

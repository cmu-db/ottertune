/*
 * OtterTune - OracleCollector.java
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
import java.sql.PreparedStatement;
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

/** */
public class OracleCollector extends DBCollector {
  private static final Logger LOG = Logger.getLogger(MySQLCollector.class);

  private static final String VERSION_SQL = "select VERSION from product_component_version";

  private static final String PARAMETERS_SQL = "select name, value from v$parameter";

  private static final String METRICS_SQL = "select metric_name, value from v$sysmetric";

  public OracleCollector(String oriDBUrl, String username, String password) {
    try {
	//System.out.println(oriDBUrl);
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
        dbParameters.put(out.getString(1).toLowerCase(), out.getString(2));
      }

      // Collect DBMS internal metrics
      out = s.executeQuery(METRICS_SQL);
      while (out.next()) {
        dbMetrics.put(out.getString(1).toLowerCase(), out.getString(2));
      }
      conn.close();
    } catch (SQLException e) {
      LOG.error("Error while collecting DB parameters: " + e.getMessage());
      e.printStackTrace();
    }
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

  @Override
  public String collectMetrics() {
    JSONStringer stringer = new JSONStringer();
    try {
      stringer.object();
      stringer.key(JSON_GLOBAL_KEY);
      JSONObject jobGlobal = new JSONObject();
      JSONObject job = new JSONObject();
      for (Map.Entry<String, String> entry : dbMetrics.entrySet()) {
        job.put(entry.getKey(), entry.getValue());
      }
      // "global" is a a placeholder
      jobGlobal.put("global", job);
      stringer.value(jobGlobal);
      stringer.key(JSON_LOCAL_KEY);
      stringer.value(null);
      stringer.endObject();
    } catch (JSONException e) {
      e.printStackTrace();
    }
    return JSONUtil.format(stringer.toString());
  }
}

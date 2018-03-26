/*
 * OtterTune - AbstractJSONValidationTestCase.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller.collectors;

import com.controller.types.JSONSchemaType;
import com.controller.util.FileUtil;
import java.io.File;
import junit.framework.TestCase;

public abstract class AbstractJSONValidationTestCase extends TestCase {

  private static final String SAMPLE_OUTPUT_PATH = "sample_output";
  private static final String SAMPLE_CONFIG_PATH = "config";

  protected String dbName;

  protected void setUp(String dbName) throws Exception {
    super.setUp();
    this.dbName = dbName;
  }

  public void testJsonKnobs() {
    String jsonKnobsPath = FileUtil.joinPath(SAMPLE_OUTPUT_PATH, this.dbName, "knobs.json");
    assertTrue(JSONSchemaType.isValidJson(JSONSchemaType.OUTPUT, new File(jsonKnobsPath)));
  }

  public void testJsonMetrics() {
    String jsonMetricsBeforePath =
        FileUtil.joinPath(SAMPLE_OUTPUT_PATH, this.dbName, "metrics_before.json");
    String jsonMetricsAfterPath =
        FileUtil.joinPath(SAMPLE_OUTPUT_PATH, this.dbName, "metrics_after.json");
    assertTrue(JSONSchemaType.isValidJson(JSONSchemaType.OUTPUT, new File(jsonMetricsBeforePath)));
    assertTrue(JSONSchemaType.isValidJson(JSONSchemaType.OUTPUT, new File(jsonMetricsAfterPath)));
  }

  public void testJsonSummary() {
    String jsonSummaryPath = FileUtil.joinPath(SAMPLE_OUTPUT_PATH, this.dbName, "summary.json");
    assertTrue(JSONSchemaType.isValidJson(JSONSchemaType.SUMMARY, new File(jsonSummaryPath)));
  }

  public void testJsonConfig() {
    String jsonConfigPath =
        FileUtil.joinPath(SAMPLE_CONFIG_PATH, "sample_" + this.dbName + "_config.json");
    assertTrue(JSONSchemaType.isValidJson(JSONSchemaType.CONFIG, new File(jsonConfigPath)));
  }
}

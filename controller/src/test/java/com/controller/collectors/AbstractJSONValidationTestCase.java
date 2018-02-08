package com.controller.collectors;

import com.github.fge.jsonschema.core.exceptions.ProcessingException;
import java.io.File;
import java.io.IOException;

import com.controller.util.FileUtil;
import com.controller.util.ValidationUtils;

import junit.framework.TestCase;

public abstract class AbstractJSONValidationTestCase extends TestCase {

  private static final String SAMPLE_OUTPUT_PATH = "sample_output";
  private static final String SAMPLE_CONFIG_PATH = "config";
  private static final String JSON_SCHEMA_PATH = "src/main/java/com/controller/json_validation_schema";

  protected String dbName;
  protected File jsonSchemaFile;
  protected File jsonSummarySchemaFile;
  protected File jsonConfigSchemaFile;

  protected void setUp(String dbName) throws Exception {
    super.setUp();
    this.dbName = dbName;
    this.jsonSchemaFile = new File(FileUtil.joinPath(JSON_SCHEMA_PATH, "schema.json"));
    this.jsonSummarySchemaFile = new File(FileUtil.joinPath(JSON_SCHEMA_PATH, "summary_schema.json"));
    this.jsonConfigSchemaFile = new File(FileUtil.joinPath(JSON_SCHEMA_PATH, "config_schema.json"));
  }

  public void testJsonKnobs() throws IOException, ProcessingException {
    String jsonKnobsPath = FileUtil.joinPath(SAMPLE_OUTPUT_PATH, this.dbName, "knobs.json");
    File jsonKnobsFile = new File(jsonKnobsPath);
    assertTrue(ValidationUtils.isJsonValid(this.jsonSchemaFile, jsonKnobsFile));
  }

  public void testJsonMetrics() throws IOException, ProcessingException {
    String jsonMetricsBeforePath = FileUtil.joinPath(SAMPLE_OUTPUT_PATH, this.dbName, "metrics_before.json");
    String jsonMetricsAfterPath = FileUtil.joinPath(SAMPLE_OUTPUT_PATH, this.dbName, "metrics_after.json");
    File jsonMetricsBeforeFile = new File(jsonMetricsBeforePath);
    File jsonMetricsAfterFile = new File(jsonMetricsAfterPath);
    assertTrue(ValidationUtils.isJsonValid(this.jsonSchemaFile, jsonMetricsBeforeFile));
    assertTrue(ValidationUtils.isJsonValid(this.jsonSchemaFile, jsonMetricsAfterFile));
  }

  public void testJsonSummary() throws IOException, ProcessingException {
    String jsonSummaryPath = FileUtil.joinPath(SAMPLE_OUTPUT_PATH, this.dbName, "summary.json");
    File jsonSummaryFile = new File(jsonSummaryPath);
    assertTrue(ValidationUtils.isJsonValid(this.jsonSummarySchemaFile, jsonSummaryFile));
  }

  public void testJsonConfig() throws IOException, ProcessingException {
    String jsonConfigPath = FileUtil.joinPath(SAMPLE_CONFIG_PATH, "sample_" + this.dbName + "_config.json");
    File jsonConfigFile = new File(jsonConfigPath);
    assertTrue(ValidationUtils.isJsonValid(this.jsonConfigSchemaFile, jsonConfigFile));
  }
}

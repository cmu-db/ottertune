/*
 * OtterTune - JSONSchemaType.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller.types;

import com.controller.util.FileUtil;
import com.controller.util.ValidationUtils;
import com.fasterxml.jackson.databind.JsonNode;
import com.github.fge.jsonschema.core.exceptions.ProcessingException;
import com.github.fge.jsonschema.main.JsonSchema;
import java.io.File;
import java.io.IOException;

public enum JSONSchemaType {

  /** Parameters: (1) schema filename */
  OUTPUT("schema.json"),
  CONFIG("config_schema.json"),
  SUMMARY("summary_schema.json");

  // Path to JSON schema directory
  private static final String SCHEMA_PATH = "src/main/java/com/controller/json_validation_schema";

  private final JsonSchema schema;

  private JSONSchemaType(String fileName) {
    JsonSchema newSchema = null;
    String configPath = FileUtil.joinPath(SCHEMA_PATH, fileName);
    try {
      newSchema = ValidationUtils.getSchemaNode(new File(configPath));
    } catch (IOException | ProcessingException e) {
      e.printStackTrace();
    }
    this.schema = newSchema;
  }

  public JsonSchema getSchema() {
    return this.schema;
  }

  public static boolean isValidJson(JSONSchemaType schemaType, String jsonString) {
    try {
      JsonNode jsonNode = ValidationUtils.getJsonNode(jsonString);
      return ValidationUtils.isJsonValid(schemaType.getSchema(), jsonNode);
    } catch (IOException | ProcessingException e) {
      e.printStackTrace();
    }
    return false;
  }

  public static boolean isValidJson(JSONSchemaType schemaType, File jsonFile) {
    try {
      JsonNode jsonNode = ValidationUtils.getJsonNode(jsonFile);
      return ValidationUtils.isJsonValid(schemaType.getSchema(), jsonNode);
    } catch (IOException | ProcessingException e) {
      e.printStackTrace();
    }
    return false;
  }
}

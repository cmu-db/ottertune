package com.controller.collectors;

import java.io.File;
import java.io.IOException;

import com.controller.util.FileUtil;
import com.controller.util.ValidationUtils;

import junit.framework.TestCase;
import com.github.fge.jsonschema.main.JsonSchema;
import com.github.fge.jsonschema.core.exceptions.ProcessingException;
import com.fasterxml.jackson.databind.JsonNode;

public class TestInvalidJSON extends TestCase {

  private static final String JSON_SCHEMA_PATH = "src/main/java/com/controller/json_validation_schema";

  // Wrong number of levels for "global"
  private static final String BAD_JSON_TEXT_1 = 
      "{" +
      "  \"global\" : {" +
      "    \"global\" : {" +
      "      \"auto_generate_certs\": {" +
      "        \"auto_pram\" : \"NO\"" +
      "      }" +
      "    }" +
      "  }," +
      "  \"local\" : {" +
      "  }" +
      "}";

  // Lacking "local"
  private static final String BAD_JSON_TEXT_2 =
    "{" +
    "  \"global\" : {" +
    "    \"global1\" : {" +
    "      \"auto_generate_certs\": \"ON\"" +
    "    }" +
    "  }" +
    "}";

  private JsonSchema jsonSchema;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    File jsonSchemaFile = new File(FileUtil.joinPath(JSON_SCHEMA_PATH, "schema.json"));
    this.jsonSchema = ValidationUtils.getSchemaNode(jsonSchemaFile);
  }

  public void testBadJSONOutput() throws IOException, ProcessingException {
    JsonNode badTextJson1 = ValidationUtils.getJsonNode(BAD_JSON_TEXT_1);
    JsonNode badTextJson2 = ValidationUtils.getJsonNode(BAD_JSON_TEXT_2);
    assertFalse(ValidationUtils.isJsonValid(this.jsonSchema, badTextJson1));
    assertFalse(ValidationUtils.isJsonValid(this.jsonSchema, badTextJson2));
  }
  
}

/*
 * OtterTune - TestInvalidJSON.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller.collectors;

import com.controller.types.JSONSchemaType;
import junit.framework.TestCase;

public class TestInvalidJSON extends TestCase {

  // Wrong number of levels for "global"
  private static final String BAD_JSON_TEXT_1 =
      "{"
          + "  \"global\" : {"
          + "    \"global\" : {"
          + "      \"auto_generate_certs\": {"
          + "        \"auto_pram\" : \"NO\""
          + "      }"
          + "    }"
          + "  },"
          + "  \"local\" : {"
          + "  }"
          + "}";

  // Lacking "local"
  private static final String BAD_JSON_TEXT_2 =
      "{"
          + "  \"global\" : {"
          + "    \"global1\" : {"
          + "      \"auto_generate_certs\": \"ON\""
          + "    }"
          + "  }"
          + "}";

  public void testBadJSONOutput() {
    assertFalse(JSONSchemaType.isValidJson(JSONSchemaType.OUTPUT, BAD_JSON_TEXT_1));
    assertFalse(JSONSchemaType.isValidJson(JSONSchemaType.OUTPUT, BAD_JSON_TEXT_2));
  }
}

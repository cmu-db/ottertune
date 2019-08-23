/*
 * OtterTune - TestMySQLJSON.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller.collectors;

public class TestMySQLJSON extends AbstractJSONValidationTestCase {

  @Override
  protected void setUp() throws Exception {
    super.setUp("mysql");
  }
}

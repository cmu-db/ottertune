/*
 * OtterTune - TestPostgresJSON.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller.collectors;

public class TestPostgresJSON extends AbstractJSONValidationTestCase {

  @Override
  protected void setUp() throws Exception {
    super.setUp("postgres");
  }
}

/*
 * OtterTune - TestOracleJSON.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller.collectors;

public class TestOracleJSON extends AbstractJSONValidationTestCase {

  @Override
  protected void setUp() throws Exception {
    super.setUp("oracle");
  }
}

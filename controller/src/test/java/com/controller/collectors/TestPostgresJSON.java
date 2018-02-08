package com.controller.collectors;

public class TestPostgresJSON extends AbstractJSONValidationTestCase {

  @Override
  protected void setUp() throws Exception {
    super.setUp("postgres");
  }

}

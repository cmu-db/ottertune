/*
 * OtterTune - DBParameterCollector.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller.collectors;

public interface DBParameterCollector {
  boolean hasParameters();

  boolean hasMetrics();

  String collectParameters();

  String collectMetrics();

  String collectVersion();
}

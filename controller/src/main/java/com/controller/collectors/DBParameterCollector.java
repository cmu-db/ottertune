package com.controller.collectors;

public interface DBParameterCollector {
    boolean hasParameters();
    boolean hasMetrics();
    String collectParameters();
    String collectMetrics();
    String collectVersion();
}

/*
 * OtterTune - Main.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller;

import com.controller.collectors.DBCollector;
import com.controller.collectors.MySQLCollector;
import com.controller.collectors.PostgresCollector;
import com.controller.collectors.SAPHanaCollector;
import com.controller.util.FileUtil;
import com.controller.util.JSONUtil;
import com.controller.util.ValidationUtils;
import com.controller.util.json.JSONException;
import com.controller.util.json.JSONObject;
import com.github.fge.jsonschema.core.exceptions.ProcessingException;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.MalformedParametersException;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
//import org.json.simple.JSONObject;

/**
 * Controller main.
 *
 * @author Shuli
 */
public class Main {
  static final Logger LOG = Logger.getLogger(Main.class);

  // Default output directory name
  private static final String DEFAULT_DIRECTORY = "output";

  // Default observation period time (5 minutes)
  private static final int DEFAULT_TIME_SECONDS = 300;

  // Path to JSON schema directory
  private static final String SCHEMA_PATH = "src/main/java/com/controller/json_validation_schema";

  private static final int TO_MILLISECONDS = 1000;

  public static void main(String[] args) {
    // Initialize log4j
    PropertyConfigurator.configure("log4j.properties");

    // Create the command line parser
    CommandLineParser parser = new PosixParser();
    Options options = new Options();
    options.addOption("c", "config", true, "[required] Controller configuration file");
    options.addOption("t", "time", true, "The observation time in seconds, default is 300");
    options.addOption(
        "d", "directory", true, "Base directory for the result files, default is 'output'");
    options.addOption("h", "help", true, "Print this help");
    String configFilePath = null;

    // Parse the command line arguments
    CommandLine argsLine;
    try {
      argsLine = parser.parse(options, args);
    } catch (ParseException e) {
      LOG.error("Unable to Parse command line arguments");
      printUsage(options);
      return;
    }

    if (argsLine.hasOption("h")) {
      printUsage(options);
      return;
    } else if (argsLine.hasOption("c") == false) {
      LOG.error("Missing configuration file");
      printUsage(options);
      return;
    }

    int time = DEFAULT_TIME_SECONDS;
    if (argsLine.hasOption("t")) {
      time = Integer.parseInt(argsLine.getOptionValue("t"));
    }
    LOG.info("Experiment time is set to: " + time);

    String outputDirectory = DEFAULT_DIRECTORY;
    if (argsLine.hasOption("d")) {
      outputDirectory = argsLine.getOptionValue("d");
    }
    LOG.info("Experiment output directory is set to: " + outputDirectory);

    // Parse controller configuration file
    String configPath = argsLine.getOptionValue("c");
    File configFile = new File(configPath);
    File schemaFile = new File(FileUtil.joinPath(SCHEMA_PATH, "config_schema.json"));

    // Check config format
    try {
      if (!ValidationUtils.isJsonValid(schemaFile, configFile)) {
        LOG.error("Invalid configuration file format");
        return;
      }
    } catch (IOException | ProcessingException e) {
      LOG.error("Error parsing configuration file: " + configPath);
      e.printStackTrace();
    }

    // Load configuration file
    ControllerConfiguration config = null;
    try {
      JSONObject input = new JSONObject(FileUtil.readFile(configFile));
      config = new ControllerConfiguration(
          input.getString("database_type"), input.getString("username"),
          input.getString("password"), input.getString("database_url"),
          input.getString("upload_code"), input.getString("upload_url"),
          input.getString("workload_name"));
    } catch (JSONException e) {
      e.printStackTrace();
    }

    String subDirectory = FileUtil.joinPath(outputDirectory, config.getDBName());
    FileUtil.makeDirIfNotExists(outputDirectory, subDirectory);

    DBCollector collector = getCollector(config);
    try {
      LOG.info("First collection of metrics before experiment");
      // first collection (before queries)
      PrintWriter metricsWriter =
          new PrintWriter(FileUtil.joinPath(subDirectory, "metrics_before.json"), "UTF-8");
      metricsWriter.println(collector.collectMetrics());
      metricsWriter.close();
      PrintWriter knobsWriter =
          new PrintWriter(FileUtil.joinPath(subDirectory, "knobs.json"), "UTF-8");
      knobsWriter.println(collector.collectParameters());
      knobsWriter.close();

      // record start time
      long startTime = System.currentTimeMillis();
      LOG.info("Starting the experiment ...");

      // go to sleep
      Thread.sleep(time * TO_MILLISECONDS);
      long endTime = System.currentTimeMillis();
      LOG.info("Done running the experiment");

      // summary json obj
      JSONObject summary = null;
      try {
        summary = new JSONObject();
        summary.put("start_time", startTime);
        summary.put("end_time", endTime);
        summary.put("observation_time", time);
        summary.put("database_type", config.getDBName());
        summary.put("database_version", collector.collectVersion());
        summary.put("workload_name", config.getWorkloadName());
      } catch (JSONException e) {
        e.printStackTrace();
      }

      // write summary JSONObject into a JSON file
      PrintWriter summaryout =
          new PrintWriter(FileUtil.joinPath(subDirectory, "summary.json"), "UTF-8");
      summaryout.println(JSONUtil.format(summary.toString()));
      summaryout.close();

      // second collection (after workload execution)
      LOG.info("Second collection of metrics after experiment");
      collector = getCollector(config);
      PrintWriter metricsWriterFinal =
          new PrintWriter(FileUtil.joinPath(subDirectory, "metrics_after.json"), "UTF-8");
      metricsWriterFinal.println(collector.collectMetrics());
      metricsWriterFinal.close();
    } catch (FileNotFoundException | UnsupportedEncodingException | InterruptedException e) {
      LOG.error("Failed to produce output files");
      e.printStackTrace();
    }

    Map<String, String> outfiles = new HashMap<>();
    outfiles.put("knobs", FileUtil.joinPath(subDirectory, "knobs.json"));
    outfiles.put("metrics_before", FileUtil.joinPath(subDirectory, "metrics_before.json"));
    outfiles.put("metrics_after", FileUtil.joinPath(subDirectory, "metrics_after.json"));
    outfiles.put("summary", FileUtil.joinPath(subDirectory, "summary.json"));
    ResultUploader.upload(
        config.getUploadURL(), config.getUploadCode(), outfiles);
  }

  private static void printUsage(Options options) {
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp("controller", options);
  }

  private static DBCollector getCollector(ControllerConfiguration config) {
    DBCollector collector = null;
    switch (config.getDBType()) {
      case POSTGRES:
        collector =
            new PostgresCollector(
                config.getDBURL(), config.getDBUsername(), config.getDBPassword());
        break;
      case MYSQL:
        collector =
            new MySQLCollector(
                config.getDBURL(), config.getDBUsername(), config.getDBPassword());
        break;
      case SAPHANA:
        collector =
            new SAPHanaCollector(
                config.getDBURL(), config.getDBUsername(), config.getDBPassword());
        break;
      default:
        LOG.error("Invalid database type");
        throw new MalformedParametersException("Invalid database type");
    }
    return collector;
  }
}

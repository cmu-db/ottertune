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
import com.controller.util.JSONUtil;
import java.io.File;
import java.io.FileNotFoundException;
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
import org.json.simple.JSONObject;

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
      LOG.info("Experiment time is set to: " + time);
    }

    String outputDirectory = DEFAULT_DIRECTORY;
    if (argsLine.hasOption("o")) {
      outputDirectory = argsLine.getOptionValue("o");
      LOG.info("Experiment output directory is set to: " + outputDirectory);
    }

    // Parse controller configuration file
    String configFile = argsLine.getOptionValue("c");
    HashMap<String, String> input = ConfigFileParser.getInputFromConfigFile(configFile);
    ControllerConfiguration controllerConfiguration =
        new ControllerConfiguration(
            DatabaseType.get(input.get("database_type")),
            input.get("username"),
            input.get("password"),
            input.get("database_url"),
            input.get("upload_code"),
            input.get("upload_url"),
            input.get("workload_name"));

    DBCollector collector = getCollector(controllerConfiguration);
    String outputDir = input.get("database_type");
    String dbtype = input.get("database_type");

    new File(outputDirectory).mkdir();
    new File(outputDirectory + "/" + outputDir).mkdir();

    try {
      // summary json obj
      JSONObject summary = new JSONObject();
      summary.put("observation_time", time);
      summary.put("database_type", dbtype);
      summary.put("database_version", collector.collectVersion());

      LOG.info("First collection of metrics before experiment");
      // first collection (before queries)
      PrintWriter metricsWriter =
          new PrintWriter(outputDirectory + "/" + outputDir + "/metrics_before.json", "UTF-8");
      metricsWriter.println(collector.collectMetrics());
      metricsWriter.flush();
      metricsWriter.close();
      PrintWriter knobsWriter =
          new PrintWriter(outputDirectory + "/" + outputDir + "/knobs.json", "UTF-8");
      knobsWriter.println(collector.collectParameters());
      knobsWriter.flush();
      knobsWriter.close();

      // record start time
      summary.put("start_time", System.currentTimeMillis());
      LOG.info("Start the experiment ...");

      // go to sleep
      Thread.sleep(time * TO_MILLISECONDS);
      LOG.info("Experiment ends");

      // record end time
      summary.put("end_time", System.currentTimeMillis());

      // record workload_name
      summary.put("workload_name", controllerConfiguration.getWorkloadName());

      // write summary JSONObject into a JSON file
      PrintWriter summaryout =
          new PrintWriter(outputDirectory + "/" + outputDir + "/summary.json", "UTF-8");
      summaryout.println(JSONUtil.format(summary.toString()));
      summaryout.flush();
      summaryout.close();

      LOG.info("Second collection of metrics after experiment");
      // second collection (after queries)
      PrintWriter metricsWriterFinal =
          new PrintWriter(outputDirectory + "/" + outputDir + "/metrics_after.json", "UTF-8");
      metricsWriterFinal.println(collector.collectMetrics());
      metricsWriterFinal.flush();
      metricsWriterFinal.close();
    } catch (FileNotFoundException | UnsupportedEncodingException | InterruptedException e) {
      LOG.error("Failed to produce output files");
      e.printStackTrace();
    }

    Map<String, String> outfiles = new HashMap<>();
    outfiles.put("knobs", outputDirectory + "/" + outputDir + "/knobs.json");
    outfiles.put("metrics_before", outputDirectory + "/" + outputDir + "/metrics_before.json");
    outfiles.put("metrics_after", outputDirectory + "/" + outputDir + "metrics_after.json");
    outfiles.put("summary", outputDirectory + "/" + outputDir + "summary.json");
    ResultUploader.upload(
        controllerConfiguration.getUploadURL(), controllerConfiguration.getUploadCode(), outfiles);
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
                config.getDBDriver(), config.getDBUsername(), config.getDBPassword());
        break;
      case MYSQL:
        collector =
            new MySQLCollector(
                config.getDBDriver(), config.getDBUsername(), config.getDBPassword());
        break;
      case SAPHANA:
        collector =
            new SAPHanaCollector(
                config.getDBDriver(), config.getDBUsername(), config.getDBPassword());
        break;
      default:
        LOG.error("Invalid database type");
        throw new MalformedParametersException("Invalid database type");
    }
    return collector;
  }
}

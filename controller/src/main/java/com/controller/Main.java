package com.controller;

import com.controller.collectors.DBCollector;
import com.controller.collectors.PostgresCollector;
import com.controller.util.JSONUtil;
import org.json.simple.JSONObject;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.Map;

/**
 * Controller main.
 * @author Shuli
 */
public class Main {
    private static final int DEFAULT_TIME = 5;  //default observation time: 5 s
    private static final int TO_MILLISECONDS = 1000;
    public static void main(String[] args) {
        // Parse command line argument
        if(args.length % 2 != 0) {
            System.out.println("Command line argument mal formed");
            return;
        }
        int time = DEFAULT_TIME; // set time to default
        for(int i = 0; i < args.length; i += 2){
            String flag = args[i];
            String val = args[++i];
            switch (flag) {
                case "-t" :
                    time = Integer.valueOf(val);
                    break;
                default:
                    System.out.println("Invalid flag: " + flag);
                    break;
            }
        }

        // Parse input config file
        String configFileName = "input_config.json";
        HashMap<String, String> input = ConfigFileParser.getInputFromConfigFile(configFileName);
        // db type
        String dbtype = input.get("database_type");
        DBCollector collector = null;
        // parameters for creating a collector.
        String username = input.get("username");
        String password = input.get("password");
        String dbURL = input.get("database_url");
        // uploader
        String uploadCode = input.get("upload_code");
        String uploadURL = input.get("upload_url");

        switch (dbtype) {
            case "postgres":
                collector = new PostgresCollector(dbURL, username, password);
                break;
            default:
                System.out.println("Please specify database type!");
                return;
        }

        try {
            // summary json obj
            JSONObject summary = new JSONObject();
            summary.put("observation_time", time);
            summary.put("database_type", dbtype);
            summary.put("database_version", collector.collectVersion());

            // first collection (before queries)
            PrintWriter metricsWriter = new PrintWriter("output/metrics_before.json", "UTF-8");
            metricsWriter.println(collector.collectMetrics());
            metricsWriter.flush();
            metricsWriter.close();
            PrintWriter knobsWriter = new PrintWriter("output/knobs.json", "UTF-8");
            knobsWriter.println(collector.collectParameters());
            knobsWriter.flush();
            knobsWriter.close();

            // record start time
            summary.put("start_time", System.currentTimeMillis());

            // go to sleep
            Thread.sleep(time * TO_MILLISECONDS);

            // record end time
            summary.put("end_time", System.currentTimeMillis());

            // write summary JSONObject into a JSON file
            PrintWriter summaryout = new PrintWriter("output/summary.json","UTF-8");
            summaryout.println(JSONUtil.format(summary.toString()));
            summaryout.flush();

            // second collection (after queries)
            PrintWriter metricsWriterFinal = new PrintWriter("output/metrics_after.json", "UTF-8");
            metricsWriterFinal.println(collector.collectMetrics());
            metricsWriterFinal.flush();
            metricsWriterFinal.close();
        } catch (FileNotFoundException | UnsupportedEncodingException | InterruptedException e) {
            e.printStackTrace();
        }

        Map<String, String> outfiles = new HashMap<>();
        outfiles.put("knobs", "output/knobs.json");
        outfiles.put("metrics_before", "output/metrics_before.json");
        outfiles.put("metrics_after", "output/metrics_after.json");
        outfiles.put("summary", "output/summary.json");
        ResultUploader.upload(uploadURL, uploadCode, outfiles);

    }
}

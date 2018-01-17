package com.controller;

import com.controller.collectors.DBCollector;
import com.controller.collectors.MySQLCollector;
import com.controller.collectors.PostgresCollector;
import com.controller.util.JSONUtil;
import org.json.simple.JSONObject;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.MalformedParametersException;
import java.util.HashMap;

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
            throw new MalformedParametersException("Command line argument is malformed");
        }
        int time = DEFAULT_TIME; // set time to default
        String configFileName = "input_config.json"; //default config file name
        for(int i = 0; i < args.length; i += 2){
            String flag = args[i];
            String val = args[++i];
            switch (flag) {
                case "-t" :
                    time = Integer.valueOf(val);
                    if(time < 0) {
                        System.out.println("Invalid time would be ignored");
                        time = DEFAULT_TIME;
                    }
                    break;
                case "-f" :
                    configFileName = val;
                    break;
                default:
                    throw new MalformedParametersException("invalid flag");
            }
        }

        // Parse input config file
        HashMap<String, String> input = ConfigFileParser.getInputFromConfigFile(configFileName);
        // db type
        String dbtype = input.get("database_type");
        System.out.println(dbtype);
        DBCollector collector = null;
        // parameters for creating a collector.
        String username = input.get("username");
        String password = input.get("password");
        String dbURL = input.get("database_url");
        // uploader
        String uploadCode = input.get("upload_code");
        String uploadURL = input.get("upload_url");
        // workload
        String workloadName = input.get("workload_name");

        switch (dbtype) {
            case "postgres":
                collector = new PostgresCollector(dbURL, username, password);
                break;
            case "mysql":
                collector = new MySQLCollector(dbURL, username, password);
                break;
            default:
                throw new MalformedParametersException("invalid database type");
        }
        String outputDir = dbtype;
        new File("output/" + outputDir).mkdir();

        try {
            // summary json obj
            JSONObject summary = new JSONObject();
            summary.put("observation_time", time);
            summary.put("database_type", dbtype);
            summary.put("database_version", collector.collectVersion());

            // first collection (before queries)
            PrintWriter metricsWriter = new PrintWriter("output/" + outputDir+ "/metrics_before.json", "UTF-8");
            metricsWriter.println(collector.collectMetrics());
            metricsWriter.flush();
            metricsWriter.close();
            PrintWriter knobsWriter = new PrintWriter("output/"+ outputDir + "/knobs.json", "UTF-8");
            knobsWriter.println(collector.collectParameters());
            knobsWriter.flush();
            knobsWriter.close();

            // record start time
            summary.put("start_time", System.currentTimeMillis());

            // go to sleep
            Thread.sleep(time * TO_MILLISECONDS);

            // record end time
            summary.put("end_time", System.currentTimeMillis());

            // record workload_name
            summary.put("workload_name", workloadName);

            // write summary JSONObject into a JSON file
            PrintWriter summaryout = new PrintWriter("output/" + outputDir + "/summary.json","UTF-8");
            summaryout.println(JSONUtil.format(summary.toString()));
            summaryout.flush();

            // second collection (after queries)
            PrintWriter metricsWriterFinal = new PrintWriter("output/" + outputDir + "/metrics_after.json", "UTF-8");
            metricsWriterFinal.println(collector.collectMetrics());
            metricsWriterFinal.flush();
            metricsWriterFinal.close();
        } catch (FileNotFoundException | UnsupportedEncodingException | InterruptedException e) {
            e.printStackTrace();
        }

//        Map<String, String> outfiles = new HashMap<>();
//        outfiles.put("knobs", "output/" + outputDir + "/knobs.json");
//        outfiles.put("metrics_before", "output/"+ outputDir + "/metrics_before.json");
//        outfiles.put("metrics_after", "output/"+outputDir+"metrics_after.json");
//        outfiles.put("summary", "output/"+outputDir+"summary.json");
//        ResultUploader.upload(uploadURL, uploadCode, outfiles);

    }
}

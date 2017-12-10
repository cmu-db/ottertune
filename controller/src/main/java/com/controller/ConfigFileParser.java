package com.controller;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

/**
 * Parsing input config JSON file.
 * @author Shuli
 */
public class ConfigFileParser {
    public static HashMap<String, String> getInputFromConfigFile(String configFileName) {
        JSONParser jsonParser = new JSONParser();
        HashMap<String, String> input = new HashMap<>();
        try {
            Object inputObj = jsonParser.parse(new FileReader(configFileName));
            JSONObject inputJSONObj = (JSONObject) inputObj;
            for(Object key : inputJSONObj.keySet()) {
                String jsonKey = (String) key;
                String jsonVal = (String) inputJSONObj.get(jsonKey);
                input.put(jsonKey, jsonVal);
            }

        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
        return input;
    }
}

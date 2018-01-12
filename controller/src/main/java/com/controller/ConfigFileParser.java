package com.controller;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.Map;

/**
 * Parsing input config JSON file.
 * @author Shuli
 */
public class ConfigFileParser {
    public static Map<String, String> getInputFromConfigFile(String configFileName) {
        ObjectMapper mapper = new ObjectMapper();
        Map<String, String> map = null;
        try {
            map = mapper.readValue(new File(configFileName),
                                   new TypeReference<Map<String, String>>(){});
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        return map;
    }
}

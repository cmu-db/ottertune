package com.controller;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Map;

/**
 * Parsing input config JSON file.
 * @author Shuli
 */
public class ConfigFileParser {

    private static final Logger LOG = LogManager.getLogger(ConfigFileParser.class);

    public static Map<String, String> getInputFromConfigFile(String configFileName) {
        LOG.debug("Reading file {}", configFileName);
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

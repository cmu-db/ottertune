package com.controller;

import org.junit.Test;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;

import static org.junit.Assert.*;

public class ConfigFileParserTest {

    @Test
    public void testValidConfigFile() {
        Map<String, String> res = ConfigFileParser.getInputFromConfigFile("input_config.json");

        assertFalse(res.isEmpty());

        assertEquals("postgres", res.get("database_type"));
        assertEquals("jdbc:postgresql://localhost:5432/postgres", res.get("database_url"));
        assertEquals("MacadamiaKitten", res.get("username"));
        assertEquals("", res.get("password"));
        assertEquals("MY_UPLOAD_CODE", res.get("upload_code"));
        assertEquals("http://localhost:8080", res.get("upload_url"));

        assertTrue(res.size() == 6);
    }

    @Test(expected = NullPointerException.class)
    public void testNullConfigFile() {
        ConfigFileParser.getInputFromConfigFile(null);
    }

    @Test
    public void testInexistentConfigFile() {
        Map<String, String> res = ConfigFileParser.getInputFromConfigFile("__noconfig_.json");
        assertNull(res);
    }

}

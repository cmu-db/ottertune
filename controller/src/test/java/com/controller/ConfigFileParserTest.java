package com.controller;

import org.json.simple.JSONObject;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;

import static org.junit.Assert.assertTrue;

/**
 * Test for ConfigFileParserTest
 * @author Shuli
 */
public class ConfigFileParserTest {
    @Before
    public void createMockJSON() {
        JSONObject mockObj = new JSONObject();
        mockObj.put("database_type", "abc");
        mockObj.put("username", "root");
        mockObj.put("password", "12345");
        mockObj.put("database_url", "jdbc:mysql://localhost:3306/mysqldb");
        try {
            PrintWriter mockJSON = new PrintWriter("./mock_config1.json","UTF-8");
            mockJSON.println(mockObj.toString());
            mockJSON.flush();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }

    }

    @Test
    public void testMockJSON() {
        HashMap<String, String> mockRes = ConfigFileParser.getInputFromConfigFile("./mock_config1.json");
        assertTrue(mockRes.get("database_type").equals("abc"));
        assertTrue(mockRes.get("username").equals("root"));
        assertTrue(mockRes.get("password").equals("12345"));
        assertTrue(mockRes.get("database_url").equals("jdbc:mysql://localhost:3306/mysqldb"));
    }

    @After
    public void deleteMockJSON() {
        File file = new File("./mock_config1.json");
        file.delete();
    }
}
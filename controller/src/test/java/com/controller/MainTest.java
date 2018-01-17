package com.controller;

import org.junit.Test;

import java.lang.reflect.MalformedParametersException;

/**
 * Test for Main.
 * @author Shuli
 */
public class MainTest {
    @Test(expected = MalformedParametersException.class)
    public void testValidCmdLineArg1() {
        Main.main(new String[]{"-t"});
    }

    @Test(expected = MalformedParametersException.class)
    public void testValidCmdLineArg2() {
        Main.main(new String[]{"-b","20"});
    }

    @Test(expected = MalformedParametersException.class)
    public void testValidCmdLineArg3() {
        Main.main(new String[]{"-b","20","-f"});
    }

    @Test(expected = MalformedParametersException.class)
    public void testValidDB() {
        Main.main(new String[]{"-f","src/test/java/com/controller/mock_config1.json"});
    }

}
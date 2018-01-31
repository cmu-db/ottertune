package com.controller;

import org.junit.Test;

/**
 * Test for Main.
 * @author Shuli
 */
public class MainTest {
    @Test(expected = NullPointerException.class)
    public void testValidCmdLineArg1() {
        Main.main(new String[]{"-t"});
    }

    @Test(expected = NullPointerException.class)
    public void testValidCmdLineArg2() {
        Main.main(new String[]{"-b","20"});
    }

    @Test(expected = NullPointerException.class)
    public void testValidCmdLineArg3() {
        Main.main(new String[]{"-b","20","-f"});
    }

    @Test(expected = NullPointerException.class)
    public void testValidDB() {
        Main.main(new String[]{"-f","src/test/java/com/controller/mock_config1.json"});
    }

}
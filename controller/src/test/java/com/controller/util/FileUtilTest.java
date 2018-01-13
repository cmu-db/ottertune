package com.controller.util;

import org.junit.Test;

import java.io.File;
import java.io.IOException;

import static org.junit.Assert.assertEquals;

public class FileUtilTest {

    @Test
    public void testJoinPath() {
        String res = FileUtil.joinPath("a", "b", "c", "d");
        assertEquals("a/b/c/d", res);

        res = FileUtil.joinPath();
        assertEquals("", res);

        res = FileUtil.joinPath("  ", "   b", "c");
        assertEquals("b/c", res);

        res = FileUtil.joinPath(null, null);
        assertEquals("", res);
    }

}

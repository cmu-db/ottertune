/*
 * OtterTune - PathUtil.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller.util;

import java.io.IOException;

/** get absolute paths of the current working directory */
public class PathUtil {

  public static String getAbsPath(String dirPath) {
    String current = null;
    try {
      current = new java.io.File(dirPath).getCanonicalPath();
    } catch (IOException e) {
      e.printStackTrace();
    }
    return current;
  }
}

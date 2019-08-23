/*
 * OtterTune - JSONSerializable.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller.util;

import com.controller.util.json.JSONException;
import com.controller.util.json.JSONObject;
import com.controller.util.json.JSONString;
import com.controller.util.json.JSONStringer;
import java.io.IOException;

public interface JSONSerializable extends JSONString {
  public void save(String outputPath) throws IOException;

  public void load(String inputPath) throws IOException;

  public void toJSON(JSONStringer stringer) throws JSONException;

  public void fromJSON(JSONObject jsonObject) throws JSONException;
}

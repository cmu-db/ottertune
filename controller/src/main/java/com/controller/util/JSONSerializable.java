package com.controller.util;

import java.io.IOException;

import com.controller.util.json.JSONException;
import com.controller.util.json.JSONObject;
import com.controller.util.json.JSONString;
import com.controller.util.json.JSONStringer;

public interface JSONSerializable extends JSONString {
    public void save(String output_path) throws IOException;
    public void load(String input_path) throws IOException;
    public void toJSON(JSONStringer stringer) throws JSONException;
    public void fromJSON(JSONObject json_object) throws JSONException;
}

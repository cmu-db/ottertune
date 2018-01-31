/******************************************************************************
 *  Copyright 2015-2018 by OLTPBenchmark Project                              *
 *                                                                            *
 *  Licensed under the Apache License, Version 2.0 (the "License");           *
 *  you may not use this file except in compliance with the License.          *
 *  You may obtain a copy of the License at                                   *
 *                                                                            *
 *    http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                            *
 *  Unless required by applicable law or agreed to in writing, software       *
 *  distributed under the License is distributed on an "AS IS" BASIS,         *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 *  See the License for the specific language governing permissions and       *
 *  limitations under the License.                                            *
 ******************************************************************************/

package com.controller.util;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

import java.io.File;
import java.io.IOException;

/**
 * @author pavlo
 */
public abstract class JSONUtil {
    private static final Logger LOG = LogManager.getLogger(JSONUtil.class.getName());
    
    /**
     * JSON Pretty Print
     * @param json
     * @return
     * @throws JsonProcessingException
     */
    public static String format(String json) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        Object obj = mapper.readValue(json, Object.class);
        return mapper.writerWithDefaultPrettyPrinter()
                     .writeValueAsString(obj);
    }
    
    /**
     * JSON Pretty Print
     * @param <T>
     * @param object
     * @return
     */
    public static <T> String format(T object) {
        ObjectMapper mapper = new ObjectMapper();
        try {
            return mapper.writer()
                    .withDefaultPrettyPrinter()
                    .writeValueAsString(object);
        }
        catch (JsonProcessingException e) {
            e.printStackTrace();
        }
        return null;
    }
    
    /**
     * 
     * @param <T>
     * @param object
     * @return
     */
    public static <T> String toJSONString(T object) {
        ObjectMapper mapper = new ObjectMapper();
        try {
            return mapper.writeValueAsString(object);
        }
        catch (JsonProcessingException e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * Write the contents of an object out to a file on the local disk as a JSON file
     * @param <T>
     * @param object
     * @param output_path
     * @throws IOException
     */
    public static <T> void save(T object, String output_path) throws IOException {
        if (LOG.isDebugEnabled()) LOG.debug("Writing out contents of " + object.getClass().getSimpleName() + " to '" + output_path + "'");
        File f = new File(output_path);
        try {
            FileUtil.makeDirIfNotExists(f.getParent());
            FileUtil.writeStringToFile(f, format(object));
        } catch (Exception ex) {
            LOG.error("Failed to serialize the " + object.getClass().getSimpleName() + " file '" + f + "'", ex);
            throw new IOException(ex);
        }
    }
}

package com.controller;

import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.entity.mime.content.FileBody;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

import java.io.File;
import java.io.IOException;
import java.util.*;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

/**
 * Uploading the result.
 * @author Shuli
 */
public class ResultUploader {

    private static final Logger LOG = LogManager.getLogger(ResultUploader.class);

    public static void upload(String uploadURL, String uploadCode, Map<String, String> files) {

        try {
            MultipartEntityBuilder mb =  MultipartEntityBuilder.create().addTextBody("upload_code", uploadCode);

/*
            files.entrySet()
                    .stream()
                    .forEach(e -> mb.addPart(e.getKey(),
                                             new FileBody(new File(e.getValue()))));
*/

            for (Map.Entry<String, String> e : files.entrySet()) {
                String filename = e.getKey();
                String path = e.getValue();
                mb.addPart(filename, new FileBody(new File(path)));
            }

            CloseableHttpClient httpclient = HttpClients.createDefault();
            HttpPost httpPost = new HttpPost(uploadURL);

            HttpEntity reqEntity = mb.build();
            httpPost.setEntity(reqEntity);

            try (CloseableHttpResponse response = httpclient.execute(httpPost)) {
                HttpEntity resEntity = response.getEntity();
                EntityUtils.consume(resEntity);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}

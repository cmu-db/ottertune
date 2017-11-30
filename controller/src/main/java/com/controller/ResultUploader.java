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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Uploading the result.
 * @author Shuli
 */
public class ResultUploader {
    public static void upload(String uploadURL, String uploadCode, Map<String, String> files) {

        try {
            List<String> filesToSendNames = new ArrayList<>();
            List<File> filesToSend = new ArrayList<>();
            for(String fileName : files.keySet()) {
                String path = files.get(fileName);
                filesToSendNames.add(fileName);
                File f = new File(path);
                filesToSend.add(f);
            }

            CloseableHttpClient httpclient = HttpClients.createDefault();
            HttpPost httppost = new HttpPost(uploadURL);


            MultipartEntityBuilder mb =  MultipartEntityBuilder.create().addTextBody("upload_code", uploadCode);
            for(int i = 0; i < filesToSendNames.size(); i ++) {
                mb.addPart(filesToSendNames.get(i), new FileBody(filesToSend.get(i)));
            }

            HttpEntity reqEntity = mb.build();
            httppost.setEntity(reqEntity);
            CloseableHttpResponse response = httpclient.execute(httppost);
            try {
                HttpEntity resEntity = response.getEntity();
                EntityUtils.consume(resEntity);
            } finally {
                response.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}

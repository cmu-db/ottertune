package com.controller.collectors;

import com.controller.util.PathUtil;
import com.controller.util.ValidationUtils;
import com.github.fge.jsonschema.core.exceptions.ProcessingException;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.io.InvalidObjectException;

/**
 * Test for DBCollector. Test the output knob/metrics json files and the output summary json file.
 * @author Shuli
 */
public class DBCollectorTest {
    private String srcPath;
    private String outputPath;
    private String schemaFilePath;

    @Before
    public void setUp() {
        srcPath = PathUtil.getAbsPath("src");
        outputPath = PathUtil.getAbsPath("sample_output");
        schemaFilePath = srcPath + "/main/java/com/controller/json_validation_schema/schema.json";
    }

    @Test
    public void pathTest() {

    }

    @Test
    public void mysqlOutputTest() throws IOException, ProcessingException {
        String knobsJsonPath = outputPath + "/mysql/knobs.json";
        String metricsAfterJsonPath = outputPath + "/mysql/metrics_after.json";
        String metricsBeforeJsonPath = outputPath + "/mysql/metrics_before.json";
        File schemaFile = new File(schemaFilePath);
        File knobsJson = new File(knobsJsonPath);
        File metricsAfterJson = new File(metricsAfterJsonPath);
        File metricsBeforeJson = new File(metricsBeforeJsonPath);

        if(!ValidationUtils.isJsonValid(schemaFile, knobsJson)) {
            throw new InvalidObjectException("invalid knobs json output file");
        }
        if(!ValidationUtils.isJsonValid(schemaFile, metricsAfterJson)) {
            throw new InvalidObjectException("invalid metrics_after json output file");
        }
        if(!ValidationUtils.isJsonValid(schemaFile, metricsBeforeJson)) {
            throw new InvalidObjectException("invalid metrics_before json output file");
        }
    }

    @Test
    public void postgresOutputTest() throws IOException, ProcessingException {
        String knobsJsonPath = outputPath + "/postgres/knobs.json";
        String metricsAfterJsonPath = outputPath + "/postgres/metrics_after.json";
        String metricsBeforeJsonPath = outputPath + "/postgres/metrics_before.json";
        File schemaFile = new File(schemaFilePath);
        File knobsJson = new File(knobsJsonPath);
        File metricsAfterJson = new File(metricsAfterJsonPath);
        File metricsBeforeJson = new File(metricsBeforeJsonPath);

        if(!ValidationUtils.isJsonValid(schemaFile, knobsJson)) {
            throw new InvalidObjectException("invalid knobs json output file");
        }
        if(!ValidationUtils.isJsonValid(schemaFile, metricsAfterJson)) {
            throw new InvalidObjectException("invalid metrics_after json output file");
        }
        if(!ValidationUtils.isJsonValid(schemaFile, metricsBeforeJson)) {
            throw new InvalidObjectException("invalid metrics_before json output file");
        }
    }

    @Test
    public void mockJsonOutputTest() throws IOException, ProcessingException {
        String mockJsonFile1Path = srcPath + "/test/java/com/controller/collectors/mock_output/mockJsonOutput1.json";
        String mockJsonFile2Path = srcPath + "/test/java/com/controller/collectors/mock_output/mockJsonOutput2.json";
        File schemaFile = new File(schemaFilePath);
        File mockJsonFile1 = new File(mockJsonFile1Path);
        File mockJsonFile2 = new File(mockJsonFile2Path);

        // wrong number of levels for "global"
        if(ValidationUtils.isJsonValid(schemaFile, mockJsonFile1)) {
            throw new InvalidObjectException("the mock json output file should be invalid!");
        }
        // lacking "local"
        if(ValidationUtils.isJsonValid(schemaFile, mockJsonFile2)) {
            throw new InvalidObjectException("the mock json output file should be invalid!");
        }
    }

    @Test
    public void outputSummaryJsonTest() throws IOException, ProcessingException {
        String summarySchemaFilePath = srcPath + "/main/java/com/controller/json_validation_schema/summary_schema.json";
        File schemaFile = new File(summarySchemaFilePath);
        String mysqlSummaryPath = outputPath + "/mysql/summary.json";
        String postgresSummaryPath = outputPath + "/postgres/summary.json";
        File mysqlSummary = new File(mysqlSummaryPath);
        File postgresSummary = new File(postgresSummaryPath);

        if(!ValidationUtils.isJsonValid(schemaFile, mysqlSummary)) {
            throw new InvalidObjectException("mysql json output summary file is invalid!");
        }
        if(!ValidationUtils.isJsonValid(schemaFile, postgresSummary)) {
            throw new InvalidObjectException("postgres json output summary file is invalid!");
        }
    }


}
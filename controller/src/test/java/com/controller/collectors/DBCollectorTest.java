package com.controller.collectors;

import com.controller.util.ValidationUtils;
import com.github.fge.jsonschema.core.exceptions.ProcessingException;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.io.InvalidObjectException;

/**
 * Test for DBCollector. Test the output knob/metrics json files and the output summary json file.
 * @author Shuli
 */
public class DBCollectorTest {
    @Test
    public void mysqlOutputTest() throws IOException, ProcessingException {
        File schemaFile = new File("/vagrant/ottertune/controller/src/main/java/com/controller/schema.json");
        File knobsJson = new File("/vagrant/ottertune/controller/output/mysql/knobs.json");
        File metricsAfterJson = new File("/vagrant/ottertune/controller/output/mysql/metrics_after.json");
        File metricsBeforeJson = new File("/vagrant/ottertune/controller/output/mysql/metrics_before.json");

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
        File schemaFile = new File("/vagrant/ottertune/controller/src/main/java/com/controller/schema.json");
        File knobsJson = new File("/vagrant/ottertune/controller/output/postgres/knobs.json");
        File metricsAfterJson = new File("/vagrant/ottertune/controller/output/postgres/metrics_after.json");
        File metricsBeforeJson = new File("/vagrant/ottertune/controller/output/postgres/metrics_before.json");

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
        File schemaFile = new File("/vagrant/ottertune/controller/src/main/java/com/controller/schema.json");
        File mockJsonFile1 = new File("/vagrant/ottertune/controller/src/test/java/com/controller/collectors/mockJsonOutput1.json");
        File mockJsonFile2 = new File("/vagrant/ottertune/controller/src/test/java/com/controller/collectors/mockJsonOutput2.json");

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
        File schemaFile = new File("/vagrant/ottertune/controller/src/main/java/com/controller/summary_schema.json");
        File mysqlSummary = new File("/vagrant/ottertune/controller/output/mysql/summary.json");
        File postgresSummary = new File("/vagrant/ottertune/controller/output/postgres/summary.json");

        if(!ValidationUtils.isJsonValid(schemaFile, mysqlSummary)) {
            throw new InvalidObjectException("mysql json output summary file is invalid!");
        }
        if(!ValidationUtils.isJsonValid(schemaFile, postgresSummary)) {
            throw new InvalidObjectException("postgres json output summary file is invalid!");
        }
    }


}
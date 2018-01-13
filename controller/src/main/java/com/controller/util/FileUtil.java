/******************************************************************************
 *  Copyright 2015-2018 by OLTPBenchmark Project                                   *
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

import java.io.*;

import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.StringJoiner;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

/**
 * @author pavlo
 */
public abstract class FileUtil {

    private static final Logger LOG = LogManager.getLogger(FileUtil.class);

    private static final Pattern EXT_SPLIT = Pattern.compile("\\.");
    
    /**
     * Join path components
     * @param args
     * @return
     */
    public static String joinPath(String... args) {
        StringJoiner result = new StringJoiner(File.separator);
        for (String a : args) {
            if (a != null && !a.trim().isEmpty()) {
                result.add(a.trim());
            }
        }
        return result.toString();
    }

    /**
     * Given a basename for a file, find the next possible filename if this file
     * already exists. For example, if the file test.res already exists, create
     * a file called, test.1.res
     * 
     * @param basename
     * @return
     */
    public static String getNextFilename(String basename) {
        
        if (!exists(basename))
            return basename;
        
        File f = new File(basename);
        if (f.isFile()) {
            String parts[] = EXT_SPLIT.split(basename);
            
            // Check how many files already exist
            int counter = 1;
            String nextName = parts[0] + "." + counter + "." + parts[1];
            while(exists(nextName)) {
                ++counter;
                nextName = parts[0] + "." + counter + "." + parts[1];
            }
            return nextName;
        }
        

        // Should we throw instead??
        return null;
    }

    public static boolean exists(String path) {
        return Files.exists(Paths.get(path));
    }

    public static String realpath(String path) {
        String ret = null;
        try {
            File file = new File(path);
            ret = file.getCanonicalPath();
        } catch (Exception ex) {
            LOG.warn(ex);
        }
        return ret;
    }

    public static String basename(String path) {
        return new File(path).getName();
    }

    public static String getExtension(File f) {
        if (f != null && f.isFile()) {
            String parts[] = EXT_SPLIT.split(f.getName());
            if (parts.length > 1) {
                return (parts[parts.length - 1]);
            }
        }
        return null;
    }

    private static void makeDir(String filename){
        if (filename != null) {
            File f = new File(filename);
            if (!f.exists()) {
                boolean created = f.mkdirs();
                if (!created){
                    throw new RuntimeException(String.format("Error while creating directory %s",
                                                             f.getAbsolutePath()));
                }
            }
        }
    }

    /**
     * Create any directory in the list paths if it doesn't exist
     * 
     * @param paths
     */
    public static void makeDirIfNotExists(String... paths) {
        Arrays.stream(paths).forEach(FileUtil::makeDir);
    }

    /**
     * Return a File handle to a temporary file location
     * 
     * @param ext
     *            the suffix of the filename
     * @param deleteOnExit
     *            whether to delete this file after the JVM exits
     * @return
     */
    public static File getTempFile(String ext, boolean deleteOnExit) {
        return getTempFile(null, ext, deleteOnExit);
    }

    public static File getTempFile(String ext) {
        return FileUtil.getTempFile(null, ext, false);
    }

    public static File getTempFile(String prefix, String suffix, boolean deleteOnExit) {
        File tempFile;
        if (suffix != null && !suffix.startsWith("."))
            suffix = "." + suffix;
        if (prefix == null)
            prefix = "hstore"; // Wut?!

        try {
            tempFile = File.createTempFile(prefix, suffix);
            if (deleteOnExit)
                tempFile.deleteOnExit();
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
        return tempFile;
    }

    /**
     * Unsafely create a temporary directory Yes I said that this was unsafe. I
     * don't care...
     * 
     * @return
     */
    public static File getTempDirectory() {
        final File temp = FileUtil.getTempFile(null);
        if (!(temp.delete())) {
            throw new RuntimeException("Could not delete temp file: " + temp.getAbsolutePath());
        } else if (!(temp.mkdir())) {
            throw new RuntimeException("Could not create temp directory: " + temp.getAbsolutePath());
        }
        return temp;
    }

    public static File writeStringToFile(String file_path, String content) throws IOException {
        return FileUtil.writeStringToFile(new File(file_path), content);
    }

    public static File writeStringToFile(File file, String content) throws IOException {
        FileWriter writer = new FileWriter(file);
        writer.write(content);
        writer.flush();
        writer.close();
        return file;
    }

    /**
     * Write the given string to a temporary file Will not delete the file after
     * the JVM exits
     * 
     * @param content
     * @return
     */
    public static File writeStringToTempFile(String content) {
        return writeStringToTempFile(content, "tmp", false);
    }

    /**
     * Write the given string to a temporary file with the given extension as
     * the suffix Will not delete the file after the JVM exits
     * 
     * @param content
     * @param ext
     * @return
     */
    public static File writeStringToTempFile(String content, String ext) {
        return writeStringToTempFile(content, ext, false);
    }

    /**
     * Write the given string to a temporary file with the given extension as
     * the suffix If deleteOnExit is true, then the file will be removed when
     * the JVM exits
     * 
     * @param content
     * @param ext
     * @param deleteOnExit
     * @return
     */
    public static File writeStringToTempFile(String content, String ext, boolean deleteOnExit) {
        File tempFile = FileUtil.getTempFile(ext, deleteOnExit);
        try {
            FileUtil.writeStringToFile(tempFile, content);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return tempFile;
    }

    public static String readFile(File path) {
        LOG.debug(path);
        return readFile(path.getAbsolutePath());
    }

    public static String readFile(String path) {
        String res;
        try (BufferedReader in = FileUtil.getReader(path)) {
            res = in.lines().collect(Collectors.joining("\n"));
        } catch (IOException ex) {
            throw new RuntimeException("Failed to read file contents from '" + path + "'", ex);
        }
        return res;
    }

    /**
     * Creates a BufferedReader for the given input path Can handle both gzip
     * and plain text files
     * 
     * @param path
     * @return
     * @throws IOException
     */
    public static BufferedReader getReader(String path) throws IOException {
        return FileUtil.getReader(new File(path));
    }

    /**
     * Creates a BufferedReader for the given input path Can handle both gzip
     * and plain text files
     * 
     * @param file
     * @return
     * @throws IOException
     */
    public static BufferedReader getReader(File file) throws IOException {
        if (!file.exists()) {
            throw new IOException("The file '" + file + "' does not exist");
        }

        BufferedReader in;
        if (file.getPath().endsWith(".gz")) {
            FileInputStream fin = new FileInputStream(file);
            GZIPInputStream gzis = new GZIPInputStream(fin);
            in = new BufferedReader(new InputStreamReader(gzis));
            LOG.debug("Reading in the zipped contents of '" + file.getName() + "'");
        } else {
            in = new BufferedReader(new FileReader(file));
            LOG.debug("Reading in the contents of '" + file.getName() + "'");
        }
        return in;
    }

    public static byte[] readBytesFromFile(String path) throws IOException {
        File file = new File(path);
        FileInputStream in = new FileInputStream(file);

        // Create the byte array to hold the data
        long length = file.length();
        byte[] bytes = new byte[(int) length];

        LOG.debug("Reading in the contents of '" + file.getAbsolutePath() + "'");

        // Read in the bytes
        int offset = 0;
        int numRead;
        while ((offset < bytes.length) && ((numRead = in.read(bytes, offset, bytes.length - offset)) >= 0)) {
            offset += numRead;
        } // WHILE
        if (offset < bytes.length) {
            throw new IOException("Failed to read the entire contents of '" + file.getName() + "'");
        }
        in.close();
        return bytes;
    }

    /**
     * Find the path to a directory below our current location in the source
     * tree Throws a RuntimeException if we go beyond our repository checkout
     * 
     * @param dirName
     * @return
     * @throws IOException
     */
    public static File findDirectory(String dirName) throws IOException {
        return FileUtil.find(dirName, new File(".").getCanonicalFile(), true).getCanonicalFile();
    }

    /**
     * Find the path to a directory below our current location in the source
     * tree Throws a RuntimeException if we go beyond our repository checkout
     * 
     * @param fileName
     * @return
     * @throws IOException
     */
    public static File findFile(String fileName) throws IOException {
        return (FileUtil.find(fileName, new File(".").getCanonicalFile(), false).getCanonicalFile());
    }

    private static File find(String name, final File current, boolean isdir) throws IOException {
        LOG.debug("Find Current Location = " + current);
        boolean has_svn = false;
        for (File file : current.listFiles()) {
            if (file.getCanonicalPath().endsWith(File.separator + name) && file.isDirectory() == isdir) {
                return (file);
                // Make sure that we don't go to far down...
            } else if (file.getCanonicalPath().endsWith(File.separator + ".svn")) {
                has_svn = true;
            }
        } // FOR
          // If we didn't see an .svn directory, then we went too far down
        if (!has_svn)
            throw new RuntimeException("Unable to find directory '" + name + "' [last_dir=" + current.getAbsolutePath() + "]");
        File next = new File(current.getCanonicalPath() + File.separator + "");
        return FileUtil.find(name, next, isdir);
    }

    /**
     * Returns a list of all the files in a directory whose name starts with the
     * provided prefix
     * 
     * @param dir
     * @param filePrefix
     * @return
     * @throws IOException
     */
    public static List<File> getFilesInDirectory(final File dir, final String filePrefix) {
        assert !dir.isDirectory() : "Invalid search directory path: " + dir;
        return Arrays.stream(dir.listFiles())
                .filter(x -> x.getName().startsWith(filePrefix))
                .collect(Collectors.toList());

    }
}

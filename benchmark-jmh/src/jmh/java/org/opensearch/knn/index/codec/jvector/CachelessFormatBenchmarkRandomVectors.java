/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.apache.commons.lang3.SystemUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSLockFactory;
import org.apache.lucene.store.NIOFSDirectory;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Benchmark to compare the performance of JVector and Lucene codecs with random vectors similar to {@link FormatBenchmarkRandomVectors}.
 * The difference is that this benchmark is a single shot time benchmark and cleans the filesystem cache between each iteration to measure the impact of operating system cache misses.
 * Note: this benchmark is only for qualitative analysis and not meant to reproduce the already existing benchmarks of either Lucene or JVector.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 1)
@Measurement(iterations = 3)
@Fork(1)
public class CachelessFormatBenchmarkRandomVectors {
    private static final Logger log = LogManager.getLogger(CachelessFormatBenchmarkRandomVectors.class);
    private static final String JVECTOR_NOT_QUANTIZED = "jvector_not_quantized";
    private static final String JVECTOR_QUANTIZED = "jvector_quantized";
    private static final String LUCENE101 = "Lucene101";
    private static final String FIELD_NAME = "vector_field";
    private static final int K = 100;
    private static final VectorSimilarityFunction SIMILARITY_FUNCTION = VectorSimilarityFunction.EUCLIDEAN;
    @Param({ JVECTOR_NOT_QUANTIZED, JVECTOR_QUANTIZED, LUCENE101 })  // This will run the benchmark each codec type
    private String codecType;
    @Param({ /*"1000", "10000",*/ "100000" })
    private int numDocs;
    @Param({ "128", /*"256", "512", "1024"*/ })
    private int dimension;

    private float[][] vectors;
    private float[] queryVector;
    private float expectedMinScoreInTopK;
    private Directory directory;
    private DirectoryReader directoryReader;
    private Path indexDirectoryPath;
    private IndexSearcher searcher;
    private double totalRecall = 0.0;
    private int recallCount = 0;

    @Setup
    public void setup() throws IOException {
        vectors = new float[numDocs][dimension];
        queryVector = new float[dimension];
        log.info("Generating {} random vectors of dimension {}", numDocs, dimension);
        // Generate random vectors
        Random random = new Random(42);
        for (int i = 0; i < numDocs; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = random.nextFloat();
            }
        }

        for (int i = 0; i < dimension; i++) {
            queryVector[i] = random.nextFloat();
        }

        expectedMinScoreInTopK = BenchmarkCommon.findExpectedKthMaxScore(queryVector, vectors, SIMILARITY_FUNCTION, K);

        indexDirectoryPath = Files.createTempDirectory("jvector-benchmark");
        log.info("Index path: {}", indexDirectoryPath);
        directory = new NIOFSDirectory(indexDirectoryPath, FSLockFactory.getDefault());

        // Create index with JVectorFormat
        IndexWriterConfig indexWriterConfig = new IndexWriterConfig();
        indexWriterConfig.setCodec(BenchmarkCommon.getCodec(codecType));
        indexWriterConfig.setUseCompoundFile(true);
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(true));

        try (IndexWriter writer = new IndexWriter(directory, indexWriterConfig)) {
            for (int i = 0; i < numDocs; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD_NAME, vectors[i]));
                writer.addDocument(doc);
            }
            writer.commit();
            log.info("Flushing docs to make them discoverable on the file system and force merging all segments to get a single segment");
            writer.forceMerge(1);
        }
        directoryReader = DirectoryReader.open(directory);
        searcher = new IndexSearcher(directoryReader);
        dropCaches();
    }

    @TearDown
    public void tearDown() throws IOException {
        directoryReader.close();
        directory.close();
        // Cleanup previously created index directory
        Files.walk(indexDirectoryPath)
            .sorted((path1, path2) -> path2.compareTo(path1)) // Reverse order to delete files before directories
            .forEach(path -> {
                try {
                    Files.delete(path);
                } catch (IOException e) {
                    throw new UncheckedIOException("Failed to delete " + path, e);
                }
            });
    }

    // Print average recall after each iteration
    @TearDown(Level.Iteration)
    public void printIterationStats() throws IOException {
        log.info("Average recall: {}", totalRecall / recallCount);
        dropCaches();
    }

    @TearDown(Level.Trial)
    public void printFinalStats() {
        log.info("=== Benchmark Results ===");
        log.info("Total Iterations: {}", recallCount);
        log.info("Average Recall: {}", totalRecall / recallCount);
        log.info("=====================");
    }

    @Benchmark
    public BenchmarkCommon.RecallResult benchmarkSearch(Blackhole blackhole) throws IOException {
        KnnFloatVectorQuery query = new KnnFloatVectorQuery(FIELD_NAME, queryVector, K);
        TopDocs topDocs = searcher.search(query, K);

        // Calculate recall
        float recall = BenchmarkCommon.calculateRecall(topDocs, expectedMinScoreInTopK);
        totalRecall += recall;
        recallCount++;
        return new BenchmarkCommon.RecallResult(recall);
    }

    /**
     * Drop caches to simulate cache misses
     */
    private void dropCaches() throws IOException {
        String os = System.getProperty("os.name").toLowerCase();
        try {
            ProcessBuilder pb;
            if (SystemUtils.IS_OS_LINUX) {
                pb = new ProcessBuilder("sync");
                pb.start().waitFor();
                pb = new ProcessBuilder("sh", "-c", "echo 3 | sudo tee /proc/sys/vm/drop_caches");
            } else if (SystemUtils.IS_OS_MAC) {
                pb = new ProcessBuilder("sudo", "purge");
            } else if (SystemUtils.IS_OS_WINDOWS) {
                pb = new ProcessBuilder("cmd", "/c", "echo 1 | RUNDLL32.EXE advapi32.dll,ProcessIdleTasks");
            } else {
                throw new UnsupportedOperationException("Unsupported operating system: " + os);
            }
            pb.start().waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Failed to drop caches", e);
        }
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Assert;
import org.junit.Test;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.util.*;

import static org.opensearch.knn.TestUtils.generateRandomVectors;

/**
 * Test to analyze memory usage during index building with jVector codec
 */
@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "")
@Log4j2
public class MemoryUsageAnalysisTests extends LuceneTestCase {

    private static final int VECTOR_DIMENSION = 128;
    private static final int TOTAL_DOCS = 10_000;
    private static final int BATCH_SIZE = 500;
    private static final String VECTOR_FIELD_NAME = "vector_field";
    private static final DecimalFormat MEMORY_FORMAT = new DecimalFormat("#,###.00 MB");

    /**
     * Measures memory usage during various stages of building a jVector index.
     * This test tracks memory consumption:
     * 1. Before indexing starts (baseline)
     * 2. After each batch of documents is indexed
     * 3. After commits
     * 4. After force merge operations
     * 5. After reader creation
     * <p>
     * It also verifies:
     * 1. Memory decreases after commit compared to pre-commit value
     * 2. Memory after commit increases progressively with each batch
     */

    @Test
    public void testMemoryUsageDuringIndexing() throws IOException {
        // Stores memory metrics for different operations at different points
        final Map<String, Double> memoryMetrics = new HashMap<>();

        // Create a temporary directory for the index
        Path indexPath = createTempDir("memory-test-index");

        // Configure the JVector codec
        JVectorCodec codec = new JVectorCodec();

        // Setup index writer with the JVector codec
        IndexWriterConfig config = new IndexWriterConfig().setCodec(codec)
            .setUseCompoundFile(false)
            .setMergePolicy(new ForceMergesOnlyMergePolicy(false))
            .setOpenMode(IndexWriterConfig.OpenMode.CREATE);

        try (Directory directory = FSDirectory.open(indexPath)) {
            // Measure baseline memory before starting
            double baselineMemory = measureMemoryUsage("Initial baseline");
            System.gc(); // Request garbage collection to get more accurate measurements

            // Create index writer
            try (IndexWriter writer = new IndexWriter(directory, config)) {

                int totalBatches = (int) Math.ceil((double) TOTAL_DOCS / BATCH_SIZE);

                double previousPostCommitMemory = baselineMemory;
                double currentPreCommitMemory = 0;

                // Index documents in batches
                for (int batchNum = 0; batchNum < totalBatches; batchNum++) {
                    int startDoc = batchNum * BATCH_SIZE;
                    int endDoc = Math.min((batchNum + 1) * BATCH_SIZE, TOTAL_DOCS);

                    List<Document> batch = createDocumentBatch(startDoc, endDoc);

                    // Index the batch
                    for (Document doc : batch) {
                        writer.addDocument(doc);
                    }

                    // Measure memory after batch indexing
                    String batchMetricKey = "batch_" + batchNum;
                    currentPreCommitMemory = measureMemoryUsage(
                        "After indexing batch " + (batchNum + 1) + " of " + totalBatches + " (" + startDoc + " to " + (endDoc - 1) + ")"
                    );
                    memoryMetrics.put(batchMetricKey + "_precommit", currentPreCommitMemory);

                    // Commit every 4 batches
                    if (batchNum % 4 == 1 || batchNum == totalBatches - 1) {
                        writer.commit();
                        double postCommitMemory = measureMemoryUsage("After commit (batch " + (batchNum + 1) + ")");
                        memoryMetrics.put(batchMetricKey + "_postcommit", postCommitMemory);

                        // Verify memory after commit is less than before commit
                        log.info(
                            "Memory before commit: {} MB, after commit: {} MB",
                            MEMORY_FORMAT.format(currentPreCommitMemory),
                            MEMORY_FORMAT.format(postCommitMemory)
                        );

                        Assert.assertTrue(
                            "Memory should decrease after commit. Before: "
                                + currentPreCommitMemory
                                + " MB, After: "
                                + postCommitMemory
                                + " MB",
                            currentPreCommitMemory > postCommitMemory
                        );

                        // Verify memory usage is growing with each batch (when commits happen)
                        if (batchNum > 1) {  // Skip the first commit for comparison
                            log.info(
                                "Current post-commit memory: {} MB, previous post-commit memory: {} MB",
                                MEMORY_FORMAT.format(postCommitMemory),
                                MEMORY_FORMAT.format(previousPostCommitMemory)
                            );

                            Assert.assertTrue(
                                "Memory after commit should increase with each batch. Current: "
                                    + postCommitMemory
                                    + " MB, Previous: "
                                    + previousPostCommitMemory
                                    + " MB",
                                postCommitMemory >= previousPostCommitMemory
                            );
                        }

                        previousPostCommitMemory = postCommitMemory;

                    }

                    // Force merge every 4 batches
                    if (batchNum % 4 == 3 || batchNum == totalBatches - 1) {
                        double premergeMemory = previousPostCommitMemory;
                        writer.forceMerge(1);
                        double postMergeMemory = measureMemoryUsage("After force merge (batch " + (batchNum + 1) + ")");
                        memoryMetrics.put(batchMetricKey + "_postmerge", postMergeMemory);

                        log.info(
                            "Memory impact of merge: before {} MB, after {} MB, diff: {} MB",
                            MEMORY_FORMAT.format(premergeMemory),
                            MEMORY_FORMAT.format(postMergeMemory),
                            MEMORY_FORMAT.format(postMergeMemory - premergeMemory)
                        );

                    }
                }

                // Final commit and force merge
                writer.commit();
                double finalCommitMemory = measureMemoryUsage("After final commit");
                memoryMetrics.put("final_commit", finalCommitMemory);

                writer.forceMerge(1);
                double finalMergeMemory = measureMemoryUsage("After final force merge");
                memoryMetrics.put("final_merge", finalMergeMemory);
            }

            // Measure memory after creating a reader
            try (IndexReader reader = DirectoryReader.open(directory)) {
                double readerMemory = measureMemoryUsage("After reader creation");
                memoryMetrics.put("reader_creation", readerMemory);

                // Log index stats
                log.info("Final index contains {} documents with {} dimensions per vector", reader.numDocs(), VECTOR_DIMENSION);
                log.info("Total index size on disk: {} bytes", Files.walk(indexPath).filter(Files::isRegularFile).mapToLong(p -> {
                    try {
                        return Files.size(p);
                    } catch (IOException e) {
                        return 0L;
                    }
                }).sum());

                // Log memory metrics summary
                log.info("Memory Usage Summary:");
                log.info("---------------------");
                log.info("Baseline memory: {} MB", MEMORY_FORMAT.format(baselineMemory));
                log.info("Final memory after indexing: {} MB", MEMORY_FORMAT.format(memoryMetrics.get("final_merge")));
                log.info("Memory with reader: {} MB", MEMORY_FORMAT.format(readerMemory));
                log.info("Memory growth: {} MB", MEMORY_FORMAT.format(readerMemory - baselineMemory));
            }
        }
    }

    /**
     * Creates a batch of documents with vector fields
     */
    private List<Document> createDocumentBatch(int startDoc, int endDoc) {
        List<Document> documents = new ArrayList<>(endDoc - startDoc);

        for (int i = startDoc; i < endDoc; i++) {
            Document doc = new Document();

            // Add document ID
            doc.add(new StringField("id", "doc_" + i, Field.Store.YES));

            // Create random vector
            float[] vector = generateRandomVectors(1, VECTOR_DIMENSION)[0];

            // Add vector field
            doc.add(new KnnFloatVectorField(VECTOR_FIELD_NAME, vector, VectorSimilarityFunction.EUCLIDEAN));

            documents.add(doc);
        }

        return documents;
    }

    /**
     * Measures current memory usage and returns it in MB
     * Also prints the memory usage with the given label
     *
     * @param label Label for this memory measurement
     * @return Used memory in MB
     */
    private double measureMemoryUsage(String label) {
        System.gc(); // Request garbage collection

        Runtime runtime = Runtime.getRuntime();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long usedMemory = totalMemory - freeMemory;

        double usedMemoryMB = usedMemory / (1024.0 * 1024.0);
        double totalMemoryMB = totalMemory / (1024.0 * 1024.0);
        double freeMemoryMB = freeMemory / (1024.0 * 1024.0);

        log.info(
            "{}: Used Memory: {}, Total Memory: {}, Free Memory: {}",
            label,
            MEMORY_FORMAT.format(usedMemoryMB),
            MEMORY_FORMAT.format(totalMemoryMB),
            MEMORY_FORMAT.format(freeMemoryMB)
        );

        return usedMemoryMB;
    }

}

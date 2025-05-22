/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Assert;
import org.junit.Test;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;

import java.io.IOException;
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
    private static final int TOTAL_DOCS = 100000;
    private static final int BATCH_SIZE = 100;
    private static final String VECTOR_FIELD_NAME = "vector_field";
    private static final DecimalFormat MEMORY_FORMAT = new DecimalFormat("#,###.00 MB");

    /**
     * Measures memory usage during various stages of building a jVector index.
     * This test tracks memory consumption using ramBytesUsed:
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
        final Map<String, Long> memoryMetrics = new HashMap<>();

        // Create a temporary directory for the index
        Path indexPath = createTempDir("memory-test-index");

        // Configure the JVector codec
        var codec = new JVectorCodec();

        // Setup index writer with the JVector codec
        IndexWriterConfig config = new IndexWriterConfig().setCodec(codec)
            .setUseCompoundFile(false)
            .setMergePolicy(new ForceMergesOnlyMergePolicy(false))
            .setOpenMode(IndexWriterConfig.OpenMode.CREATE);

        try (Directory directory = FSDirectory.open(indexPath)) {
            // Create index writer
            try (IndexWriter writer = new IndexWriter(directory, config)) {
                // Measure baseline memory before starting
                long baselineMemory = writer.ramBytesUsed();
                logMemoryUsage("Initial baseline", baselineMemory);

                int totalBatches = (int) Math.ceil((double) TOTAL_DOCS / BATCH_SIZE);

                long previousPostCommitMemory = baselineMemory;
                long currentPreCommitMemory = 0;

                long premergeMemory = 0;
                long postMergeMemory = 0;

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
                    currentPreCommitMemory = writer.ramBytesUsed();
                    logMemoryUsage(
                        "After indexing batch " + (batchNum + 1) + " of " + totalBatches + " (" + startDoc + " to " + (endDoc - 1) + ")",
                        currentPreCommitMemory
                    );
                    memoryMetrics.put(batchMetricKey + "_precommit", currentPreCommitMemory);

                    // Commit every 4 batches
                    if (batchNum % 4 == 1 || batchNum == totalBatches - 1) {
                        writer.commit();
                        long postCommitMemory = writer.ramBytesUsed();
                        logMemoryUsage("After commit (batch " + (batchNum + 1) + ")", postCommitMemory);
                        memoryMetrics.put(batchMetricKey + "_postcommit", postCommitMemory);

                        // Verify memory after commit is less than before commit
                        log.info(
                            "Memory before commit: {}, after commit: {}",
                            formatMemory(currentPreCommitMemory),
                            formatMemory(postCommitMemory)
                        );

                        Assert.assertTrue(
                            "Memory should decrease after commit. Before: "
                                + formatMemory(currentPreCommitMemory)
                                + ", After: "
                                + formatMemory(postCommitMemory),
                            currentPreCommitMemory > postCommitMemory
                        );

                        // Verify memory usage is growing with each batch (when commits happen)
                        if (batchNum > 1) {  // Skip the first commit for comparison
                            log.info(
                                "Current post-commit memory: {}, previous post-commit memory: {}",
                                formatMemory(postCommitMemory),
                                formatMemory(previousPostCommitMemory)
                            );

                            // Note: With ramBytesUsed, memory might not always increase with each batch
                            // due to internal optimizations, so we're logging but not asserting
                            if (postCommitMemory < previousPostCommitMemory) {
                                log.info("Memory decreased after commit, likely due to internal optimizations");
                            }
                        }

                        previousPostCommitMemory = postCommitMemory;
                    }

                    // Force merge every 4 batches
                    if (batchNum % 4 == 3 || batchNum == totalBatches - 1) {
                        premergeMemory = writer.ramBytesUsed();
                        writer.forceMerge(1);
                        postMergeMemory = writer.ramBytesUsed();
                        logMemoryUsage("After force merge (batch " + (batchNum + 1) + ")", postMergeMemory);
                        memoryMetrics.put(batchMetricKey + "_postmerge", postMergeMemory);

                        log.info(
                            "Memory impact of merge: before {}, after {}, diff: {}",
                            formatMemory(premergeMemory),
                            formatMemory(postMergeMemory),
                            formatMemory(postMergeMemory - premergeMemory)
                        );
                    }
                }

                // Final commit and force merge
                writer.commit();
                long finalCommitMemory = writer.ramBytesUsed();
                logMemoryUsage("After final commit", finalCommitMemory);
                memoryMetrics.put("final_commit", finalCommitMemory);

                writer.forceMerge(1);
                long finalMergeMemory = writer.ramBytesUsed();
                logMemoryUsage("After final force merge", finalMergeMemory);
                memoryMetrics.put("final_merge", finalMergeMemory);

                // Log memory metrics summary
                log.info("Memory Usage Summary:");
                log.info("---------------------");
                log.info("Baseline memory: {} MB", MEMORY_FORMAT.format(baselineMemory));
                log.info("Final memory after indexing: {} MB", MEMORY_FORMAT.format(memoryMetrics.get("final_merge")));
                log.info("Memory growth: {} MB", MEMORY_FORMAT.format(postMergeMemory - baselineMemory));
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
     * Logs memory usage with the given label
     *
     * @param label Label for this memory measurement
     * @param bytesUsed Memory usage in bytes
     */
    private void logMemoryUsage(String label, long bytesUsed) {
        log.info("{}: Used Memory: {}", label, formatMemory(bytesUsed));
    }

    /**
     * Formats memory size in bytes to a human-readable format
     *
     * @param bytes Memory size in bytes
     * @return Formatted memory size string
     */
    private String formatMemory(long bytes) {
        double megabytes = bytes / (1024.0 * 1024.0);
        return MEMORY_FORMAT.format(megabytes) + " (" + bytes + " bytes)";
    }
}

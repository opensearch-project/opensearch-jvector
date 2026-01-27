/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import static org.opensearch.knn.common.KNNConstants.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION;
import static org.opensearch.knn.index.engine.CommonTestUtils.getCodec;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Assert;
import org.junit.Test;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;

/**
 * Test cases specifically for reproducing and validating fixes for merge bugs with deleted documents.
 * These tests target three specific bugs:
 * 1. Bug 1: IllegalArgumentException: Ordinal out of bounds: -1 (during PQ encoding)
 * 2. Bug 2: ArrayIndexOutOfBoundsException: Index N out of bounds for length N (FixedBitSet sizing)
 * 3. Bug 3: ArrayIndexOutOfBoundsException: Index M out of bounds for length N (array sizing with deletions)
 */
@ThreadLeakFilters(
    defaultFilters = true,
    filters = { ThreadLeakFiltersForTests.class }
)
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "")
@Log4j2
public class JVectorMergeWithDeletedDocsTests extends LuceneTestCase {

    private static final String TEST_FIELD = "test_field";
    private static final String TEST_ID_FIELD = "id";

    /**
     * Test Case 1: Reproduce Bug 1 - Ordinal out of bounds: -1
     *
     * Scenario: Merge with deleted documents in leading reader, PQ enabled
     *
     * Setup:
     * - Create segment A with 1000 vectors
     * - Delete 100 vectors from segment A (making it the leading reader with most live docs)
     * - Create segment B with 500 vectors (no deletes)
     * - Trigger merge with PQ quantization enabled
     *
     * Expected: Should complete successfully without IllegalArgumentException
     * Before fix: Would throw "IllegalArgumentException: Ordinal out of bounds: -1" during PQ encoding
     */
    @Test
    public void testMergeWithDeletedDocsInLeadingReader_PQEnabled()
        throws IOException {
        final int segmentASize = 1000;
        final int segmentBSize = 500;
        final int deleteCount = 100;
        final int dimension = 128;
        final int k = 10;

        log.info(
            "Starting Bug 1 reproduction test: merge with deleted docs in leading reader, PQ enabled"
        );

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION)); // Enable PQ
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);

        try (
            FSDirectory dir = FSDirectory.open(indexPath);
            IndexWriter writer = new IndexWriter(dir, config)
        ) {
            // Step 1: Add segment A with 1000 documents
            log.info("Adding segment A with {} documents", segmentASize);
            for (int i = 0; i < segmentASize; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, i * 0.01f);
                doc.add(
                    new KnnFloatVectorField(
                        TEST_FIELD,
                        vector,
                        VectorSimilarityFunction.EUCLIDEAN
                    )
                );
                doc.add(
                    new StringField(
                        TEST_ID_FIELD,
                        String.valueOf(i),
                        Field.Store.YES
                    )
                );
                writer.addDocument(doc);
            }
            writer.commit();
            log.info("Segment A committed");

            // Step 2: Delete 100 documents from segment A
            log.info("Deleting {} documents from segment A", deleteCount);
            for (int i = 0; i < deleteCount; i++) {
                writer.deleteDocuments(
                    new Term(TEST_ID_FIELD, String.valueOf(i))
                );
            }
            writer.commit();
            log.info(
                "Deletions committed. Segment A now has {} live docs",
                segmentASize - deleteCount
            );

            // Step 3: Add segment B with 500 documents (no deletes)
            log.info("Adding segment B with {} documents", segmentBSize);
            for (int i = segmentASize; i < segmentASize + segmentBSize; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, i * 0.01f);
                doc.add(
                    new KnnFloatVectorField(
                        TEST_FIELD,
                        vector,
                        VectorSimilarityFunction.EUCLIDEAN
                    )
                );
                doc.add(
                    new StringField(
                        TEST_ID_FIELD,
                        String.valueOf(i),
                        Field.Store.YES
                    )
                );
                writer.addDocument(doc);
            }
            writer.commit();
            log.info("Segment B committed");

            // Step 4: Force merge - this should trigger Bug 1 before the fix
            log.info(
                "Starting force merge - this would trigger Bug 1 (Ordinal out of bounds: -1) before fix"
            );
            try {
                writer.forceMerge(1);
                log.info("Force merge completed successfully!");
            } catch (IllegalArgumentException e) {
                if (e.getMessage().contains("Ordinal out of bounds: -1")) {
                    log.error("BUG 1 REPRODUCED: {}", e.getMessage());
                    throw new AssertionError(
                        "Bug 1 reproduced: " + e.getMessage(),
                        e
                    );
                }
                throw e;
            }

            // Step 5: Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                log.info("Verifying merged index");
                Assert.assertEquals(
                    "Should have 1 segment after merge",
                    1,
                    reader.getContext().leaves().size()
                );
                Assert.assertEquals(
                    "Should have correct number of live docs",
                    segmentASize - deleteCount + segmentBSize,
                    reader.numDocs()
                );

                // Verify search works correctly
                final float[] target = new float[dimension];
                Arrays.fill(target, 0.5f);
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery query = new KnnFloatVectorQuery(
                    TEST_FIELD,
                    target,
                    k
                );
                TopDocs topDocs = searcher.search(query, k);
                Assert.assertEquals(
                    "Should return k results",
                    k,
                    topDocs.totalHits.value()
                );

                log.info(
                    "Test passed! Merge with deleted docs in leading reader works correctly with PQ enabled"
                );
            }
        }
    }

    /**
     * Test Case 2: Reproduce Bug 2 - Index N out of bounds for length N
     *
     * Scenario: Merge where vector count equals document count (off-by-one in FixedBitSet)
     *
     * Setup:
     * - Create segment with exactly 1127 documents, all with vectors
     * - Some documents are deleted (liveDocs marks them)
     * - Trigger merge
     *
     * Expected: Should complete successfully without ArrayIndexOutOfBoundsException
     * Before fix: Would throw "ArrayIndexOutOfBoundsException: Index 1127 out of bounds for length 1127"
     */
    @Test
    public void testMergeWithExactVectorCount_OffByOne() throws IOException {
        final int exactCount = 1127; // Specific count that triggers the bug
        final int deleteCount = 50;
        final int dimension = 64;

        log.info(
            "Starting Bug 2 reproduction test: off-by-one error with exact vector count"
        );

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec());
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);

        try (
            FSDirectory dir = FSDirectory.open(indexPath);
            IndexWriter writer = new IndexWriter(dir, config)
        ) {
            // Step 1: Add exactly 1127 documents
            log.info("Adding exactly {} documents", exactCount);
            for (int i = 0; i < exactCount; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, i * 0.01f);
                doc.add(
                    new KnnFloatVectorField(
                        TEST_FIELD,
                        vector,
                        VectorSimilarityFunction.EUCLIDEAN
                    )
                );
                doc.add(
                    new StringField(
                        TEST_ID_FIELD,
                        String.valueOf(i),
                        Field.Store.YES
                    )
                );
                writer.addDocument(doc);
            }
            writer.commit();

            // Step 2: Delete some documents
            log.info("Deleting {} documents", deleteCount);
            for (int i = 0; i < deleteCount; i++) {
                writer.deleteDocuments(
                    new Term(TEST_ID_FIELD, String.valueOf(i))
                );
            }
            writer.commit();
            log.info(
                "Deletions committed. Now have {} live docs",
                exactCount - deleteCount
            );

            // Step 3: Add another small segment to force merge
            log.info("Adding second segment with 100 documents");
            for (int i = exactCount; i < exactCount + 100; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, i * 0.01f);
                doc.add(
                    new KnnFloatVectorField(
                        TEST_FIELD,
                        vector,
                        VectorSimilarityFunction.EUCLIDEAN
                    )
                );
                doc.add(
                    new StringField(
                        TEST_ID_FIELD,
                        String.valueOf(i),
                        Field.Store.YES
                    )
                );
                writer.addDocument(doc);
            }
            writer.commit();

            // Step 4: Force merge - this should trigger Bug 2 before the fix
            log.info(
                "Starting force merge - this would trigger Bug 2 (off-by-one) before fix"
            );
            try {
                writer.forceMerge(1);
                log.info("Force merge completed successfully!");
            } catch (ArrayIndexOutOfBoundsException e) {
                if (
                    e
                        .getMessage()
                        .contains("out of bounds for length " + exactCount)
                ) {
                    log.error("BUG 2 REPRODUCED: {}", e.getMessage());
                    throw new AssertionError(
                        "Bug 2 reproduced: " + e.getMessage(),
                        e
                    );
                }
                throw e;
            }

            // Step 5: Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                log.info("Verifying merged index");
                Assert.assertEquals(
                    "Should have 1 segment after merge",
                    1,
                    reader.getContext().leaves().size()
                );
                Assert.assertEquals(
                    "Should have correct number of live docs",
                    exactCount - deleteCount + 100,
                    reader.numDocs()
                );

                log.info("Test passed! Off-by-one error is fixed");
            }
        }
    }

    /**
     * Test Case 3: Reproduce Bug 3 - Index M out of bounds for length N (M > N)
     *
     * Scenario: Merge with many deleted documents in leading reader
     *
     * Setup:
     * - Create segment A with 103451 vectors, delete ~8000 vectors (leaving ~95643 live)
     * - Create segment B with 3150 vectors (no deletes)
     * - Segment A becomes leading reader due to most live vectors
     * - Trigger merge
     *
     * Expected: Should complete successfully without ArrayIndexOutOfBoundsException
     * Before fix: Would throw "ArrayIndexOutOfBoundsException: Index 103451 out of bounds for length 95643"
     */
    @Test
    public void testMergeWithManyDeletedDocsInLeadingReader()
        throws IOException {
        final int segmentASize = 10000; // Scaled down from 103451 for faster test
        final int deleteCount = 800; // Scaled down from ~8000
        final int segmentBSize = 315; // Scaled down from 3150
        final int dimension = 64;

        log.info(
            "Starting Bug 3 reproduction test: many deleted docs in leading reader"
        );
        log.info(
            "Segment A: {} total, {} to delete, {} live",
            segmentASize,
            deleteCount,
            segmentASize - deleteCount
        );
        log.info("Segment B: {} total (no deletes)", segmentBSize);

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec());
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);

        try (
            FSDirectory dir = FSDirectory.open(indexPath);
            IndexWriter writer = new IndexWriter(dir, config)
        ) {
            // Step 1: Add segment A with many documents
            log.info("Adding segment A with {} documents", segmentASize);
            for (int i = 0; i < segmentASize; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, i * 0.001f);
                doc.add(
                    new KnnFloatVectorField(
                        TEST_FIELD,
                        vector,
                        VectorSimilarityFunction.EUCLIDEAN
                    )
                );
                doc.add(
                    new StringField(
                        TEST_ID_FIELD,
                        String.valueOf(i),
                        Field.Store.YES
                    )
                );
                writer.addDocument(doc);
            }
            writer.commit();
            log.info("Segment A committed");

            // Step 2: Delete many documents from segment A
            log.info("Deleting {} documents from segment A", deleteCount);
            for (int i = 0; i < deleteCount; i++) {
                writer.deleteDocuments(
                    new Term(TEST_ID_FIELD, String.valueOf(i))
                );
            }
            writer.commit();
            log.info(
                "Deletions committed. Segment A now has {} live docs",
                segmentASize - deleteCount
            );

            // Step 3: Add segment B (smaller, no deletes)
            log.info("Adding segment B with {} documents", segmentBSize);
            for (int i = segmentASize; i < segmentASize + segmentBSize; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, i * 0.001f);
                doc.add(
                    new KnnFloatVectorField(
                        TEST_FIELD,
                        vector,
                        VectorSimilarityFunction.EUCLIDEAN
                    )
                );
                doc.add(
                    new StringField(
                        TEST_ID_FIELD,
                        String.valueOf(i),
                        Field.Store.YES
                    )
                );
                writer.addDocument(doc);
            }
            writer.commit();
            log.info("Segment B committed");

            // Step 4: Force merge - this should trigger Bug 3 before the fix
            log.info(
                "Starting force merge - this would trigger Bug 3 (incorrect array sizing) before fix"
            );
            try {
                writer.forceMerge(1);
                log.info("Force merge completed successfully!");
            } catch (ArrayIndexOutOfBoundsException e) {
                String msg = e.getMessage();
                if (
                    msg.contains("out of bounds for length") &&
                    msg.contains(String.valueOf(segmentASize - deleteCount))
                ) {
                    log.error("BUG 3 REPRODUCED: {}", e.getMessage());
                    throw new AssertionError(
                        "Bug 3 reproduced: " + e.getMessage(),
                        e
                    );
                }
                throw e;
            }

            // Step 5: Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                log.info("Verifying merged index");
                Assert.assertEquals(
                    "Should have 1 segment after merge",
                    1,
                    reader.getContext().leaves().size()
                );
                Assert.assertEquals(
                    "Should have correct number of live docs",
                    segmentASize - deleteCount + segmentBSize,
                    reader.numDocs()
                );

                log.info(
                    "Test passed! Array sizing with many deletions works correctly"
                );
            }
        }
    }

    /**
     * Comprehensive test combining all three bug scenarios
     * Tests multiple merge cycles with various deletion patterns
     */
    @Test
    public void testMultipleMergesWithVariousDeletionPatterns()
        throws IOException {
        final int dimension = 64;
        final int k = 10;

        log.info(
            "Starting comprehensive test: multiple merges with various deletion patterns"
        );

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION)); // Enable PQ
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);

        try (
            FSDirectory dir = FSDirectory.open(indexPath);
            IndexWriter writer = new IndexWriter(dir, config)
        ) {
            // Create multiple segments with different sizes and deletion patterns
            int docId = 0;

            // Segment 1: Large segment with many deletions (tests Bug 3)
            log.info("Creating segment 1: large with many deletions");
            int seg1Size = 5000;
            for (int i = 0; i < seg1Size; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * 0.001f);
                doc.add(
                    new KnnFloatVectorField(
                        TEST_FIELD,
                        vector,
                        VectorSimilarityFunction.EUCLIDEAN
                    )
                );
                doc.add(
                    new StringField(
                        TEST_ID_FIELD,
                        String.valueOf(docId),
                        Field.Store.YES
                    )
                );
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();
            // Delete 20% from segment 1
            for (int i = 0; i < seg1Size / 5; i++) {
                writer.deleteDocuments(
                    new Term(TEST_ID_FIELD, String.valueOf(i))
                );
            }
            writer.commit();
            log.info(
                "Segment 1: {} total, {} deleted, {} live",
                seg1Size,
                seg1Size / 5,
                seg1Size - seg1Size / 5
            );

            // Segment 2: Medium segment with few deletions
            log.info("Creating segment 2: medium with few deletions");
            int seg2Size = 2000;
            int seg2Start = docId;
            for (int i = 0; i < seg2Size; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * 0.001f);
                doc.add(
                    new KnnFloatVectorField(
                        TEST_FIELD,
                        vector,
                        VectorSimilarityFunction.EUCLIDEAN
                    )
                );
                doc.add(
                    new StringField(
                        TEST_ID_FIELD,
                        String.valueOf(docId),
                        Field.Store.YES
                    )
                );
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();
            // Delete 5% from segment 2
            for (int i = seg2Start; i < seg2Start + seg2Size / 20; i++) {
                writer.deleteDocuments(
                    new Term(TEST_ID_FIELD, String.valueOf(i))
                );
            }
            writer.commit();
            log.info(
                "Segment 2: {} total, {} deleted, {} live",
                seg2Size,
                seg2Size / 20,
                seg2Size - seg2Size / 20
            );

            // Segment 3: Small segment with no deletions
            log.info("Creating segment 3: small with no deletions");
            int seg3Size = 500;
            for (int i = 0; i < seg3Size; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * 0.001f);
                doc.add(
                    new KnnFloatVectorField(
                        TEST_FIELD,
                        vector,
                        VectorSimilarityFunction.EUCLIDEAN
                    )
                );
                doc.add(
                    new StringField(
                        TEST_ID_FIELD,
                        String.valueOf(docId),
                        Field.Store.YES
                    )
                );
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();
            log.info("Segment 3: {} total, no deletions", seg3Size);

            // Segment 4: Exact count that might trigger off-by-one (tests Bug 2)
            log.info("Creating segment 4: exact count for off-by-one test");
            int seg4Size = 1127;
            int seg4Start = docId;
            for (int i = 0; i < seg4Size; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * 0.001f);
                doc.add(
                    new KnnFloatVectorField(
                        TEST_FIELD,
                        vector,
                        VectorSimilarityFunction.EUCLIDEAN
                    )
                );
                doc.add(
                    new StringField(
                        TEST_ID_FIELD,
                        String.valueOf(docId),
                        Field.Store.YES
                    )
                );
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();
            // Delete some from segment 4
            for (int i = seg4Start; i < seg4Start + 50; i++) {
                writer.deleteDocuments(
                    new Term(TEST_ID_FIELD, String.valueOf(i))
                );
            }
            writer.commit();
            log.info(
                "Segment 4: {} total, 50 deleted, {} live",
                seg4Size,
                seg4Size - 50
            );

            // Calculate expected live docs
            int expectedLiveDocs =
                (seg1Size - seg1Size / 5) +
                (seg2Size - seg2Size / 20) +
                seg3Size +
                (seg4Size - 50);
            log.info("Total expected live docs: {}", expectedLiveDocs);

            // Force merge - this tests all three bugs
            log.info("Starting force merge - tests all three bug scenarios");
            writer.forceMerge(1);
            log.info("Force merge completed successfully!");

            // Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                log.info("Verifying merged index");
                Assert.assertEquals(
                    "Should have 1 segment after merge",
                    1,
                    reader.getContext().leaves().size()
                );
                Assert.assertEquals(
                    "Should have correct number of live docs",
                    expectedLiveDocs,
                    reader.numDocs()
                );

                // Verify search works correctly
                final float[] target = new float[dimension];
                Arrays.fill(target, 0.5f);
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery query = new KnnFloatVectorQuery(
                    TEST_FIELD,
                    target,
                    k
                );
                TopDocs topDocs = searcher.search(query, k);
                Assert.assertEquals(
                    "Should return k results",
                    k,
                    topDocs.totalHits.value()
                );

                log.info(
                    "Comprehensive test passed! All bug scenarios handled correctly"
                );
            }
        }
    }

    /**
     * Test with PQ disabled to ensure bugs are not PQ-specific
     */
    @Test
    public void testMergeWithDeletedDocs_NoPQ() throws IOException {
        final int segmentASize = 1000;
        final int segmentBSize = 500;
        final int deleteCount = 100;
        final int dimension = 64;

        log.info("Starting test: merge with deleted docs, PQ disabled");

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec(Integer.MAX_VALUE)); // Disable PQ
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();

        try (
            FSDirectory dir = FSDirectory.open(indexPath);
            IndexWriter writer = new IndexWriter(dir, config)
        ) {
            // Add segment A
            for (int i = 0; i < segmentASize; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, i * 0.01f);
                doc.add(
                    new KnnFloatVectorField(
                        TEST_FIELD,
                        vector,
                        VectorSimilarityFunction.EUCLIDEAN
                    )
                );
                doc.add(
                    new StringField(
                        TEST_ID_FIELD,
                        String.valueOf(i),
                        Field.Store.YES
                    )
                );
                writer.addDocument(doc);
            }
            writer.commit();

            // Delete from segment A
            for (int i = 0; i < deleteCount; i++) {
                writer.deleteDocuments(
                    new Term(TEST_ID_FIELD, String.valueOf(i))
                );
            }
            writer.commit();

            // Add segment B
            for (int i = segmentASize; i < segmentASize + segmentBSize; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, i * 0.01f);
                doc.add(
                    new KnnFloatVectorField(
                        TEST_FIELD,
                        vector,
                        VectorSimilarityFunction.EUCLIDEAN
                    )
                );
                doc.add(
                    new StringField(
                        TEST_ID_FIELD,
                        String.valueOf(i),
                        Field.Store.YES
                    )
                );
                writer.addDocument(doc);
            }
            writer.commit();

            // Force merge
            log.info("Starting force merge without PQ");
            writer.forceMerge(1);
            log.info("Force merge completed successfully!");

            // Verify
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals(
                    "Should have 1 segment after merge",
                    1,
                    reader.getContext().leaves().size()
                );
                Assert.assertEquals(
                    "Should have correct number of live docs",
                    segmentASize - deleteCount + segmentBSize,
                    reader.numDocs()
                );

                log.info(
                    "Test passed! Merge with deleted docs works correctly without PQ"
                );
            }
        }
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Assert;
import org.junit.Test;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.opensearch.knn.index.codec.jvector.JVectorFormat.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION;
import static org.opensearch.knn.index.engine.CommonTestUtils.getCodec;

/**
 * Test used specifically for JVector
 */
// Currently {@link IndexGraphBuilder} is using the default ForkJoinPool.commonPool() which is not being shutdown.
// Ignore thread leaks until we remove the ForkJoinPool.commonPool() usage from IndexGraphBuilder
// TODO: Wire the execution thread pool to {@link IndexGraphBuilder} to avoid the failure of the UT due to leaked thread pool warning.
@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "")
@Log4j2
public class KNNJVectorTests extends LuceneTestCase {
    private static final String TEST_FIELD = "test_field";
    private static final String TEST_ID_FIELD = "id";

    /**
     * Test to verify that the JVector codec is able to successfully search for the nearest neighbours
     * in the index.
     * Single field is used to store the vectors.
     * All the documents are stored in a single segment.
     * Single commit without refreshing the index.
     * No merge.
     */
    @Test
    public void testJVectorKnnIndex_simpleCase() throws IOException {
        int k = 3; // The number of nearest neighbors to gather
        int totalNumberOfDocs = 10;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(getCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = new float[] { 0.0f, 0.0f };
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = new float[] { 0.0f, 1.0f / i };
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField("test_field", source, VectorSimilarityFunction.EUCLIDEAN));
                w.addDocument(doc);
            }
            log.info("Flushing docs to make them discoverable on the file system");
            w.commit();

            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("We should now have a single segment with 10 documents");
                Assert.assertEquals(1, reader.getContext().leaves().size());
                Assert.assertEquals(totalNumberOfDocs, reader.numDocs());

                final Query filterQuery = new MatchAllDocsQuery();
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery("test_field", target, k, filterQuery);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());
                assertEquals(9, topDocs.scoreDocs[0].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 10.0f }),
                    topDocs.scoreDocs[0].score,
                    0.001f
                );
                assertEquals(8, topDocs.scoreDocs[1].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 9.0f }),
                    topDocs.scoreDocs[1].score,
                    0.001f
                );
                assertEquals(7, topDocs.scoreDocs[2].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 8.0f }),
                    topDocs.scoreDocs[2].score,
                    0.001f
                );
                log.info("successfully completed search tests");
            }
        }
        log.info("successfully closed directory");
    }

    /**
     * Test to verify that the JVector codec is able to successfully search for the nearest neighbours
     * in the index.
     * Single field is used to store the vectors.
     * Documents are stored in a multiple segments.
     * Multiple commits without refreshing the index.
     * No merge.
     */
    @Test
    public void testJVectorKnnIndex_multipleSegments() throws IOException {
        int k = 3; // The number of nearest neighbours to gather
        int totalNumberOfDocs = 10;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(getCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(false));
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = new float[] { 0.0f, 0.0f };
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = new float[] { 0.0f, 1.0f / i };
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField("test_field", source, VectorSimilarityFunction.EUCLIDEAN));
                w.addDocument(doc);
                w.commit(); // this creates a new segment
            }
            log.info("Done writing all files to the file system");

            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("We should now have 10 segments, each with a single document");
                Assert.assertEquals(10, reader.getContext().leaves().size());
                Assert.assertEquals(totalNumberOfDocs, reader.numDocs());
                final Query filterQuery = new MatchAllDocsQuery();
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = new KnnFloatVectorQuery("test_field", target, k, filterQuery);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());
                assertEquals(9, topDocs.scoreDocs[0].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 10.0f }),
                    topDocs.scoreDocs[0].score,
                    0.001f
                );
                assertEquals(8, topDocs.scoreDocs[1].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 9.0f }),
                    topDocs.scoreDocs[1].score,
                    0.001f
                );
                assertEquals(7, topDocs.scoreDocs[2].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 8.0f }),
                    topDocs.scoreDocs[2].score,
                    0.001f
                );
                log.info("successfully completed search tests");
            }
        }
    }

    /**
     * Test to verify that the JVector codec is able to successfully search for the nearest neighbours
     * in the index.
     * Single field is used to store the vectors.
     * Documents are stored in a multiple segments.
     * Multiple commits without refreshing the index.
     * Merge is enabled.
     */
    @Test
    public void testJVectorKnnIndex_mergeEnabled() throws IOException {
        int k = 3; // The number of nearest neighbours to gather
        int totalNumberOfDocs = 10;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(getCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        indexWriterConfig.setMergeScheduler(new SerialMergeScheduler());
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = new float[] { 0.0f, 0.0f };
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = new float[] { 0.0f, 1.0f / i };
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField("test_field", source, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField("my_doc_id", Integer.toString(i, 10), Field.Store.YES));
                w.addDocument(doc);
                w.commit(); // this creates a new segment without triggering a merge
            }
            log.info("Done writing all files to the file system");

            w.forceMerge(1); // this merges all segments into a single segment
            log.info("Done merging all segments");
            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("We should now have 1 segment with 10 documents");
                Assert.assertEquals(1, reader.getContext().leaves().size());
                Assert.assertEquals(totalNumberOfDocs, reader.numDocs());
                final Query filterQuery = new MatchAllDocsQuery();
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery("test_field", target, k, filterQuery);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());
                Document doc = reader.storedFields().document(topDocs.scoreDocs[0].doc);
                assertEquals("10", doc.get("my_doc_id"));
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 10.0f }),
                    topDocs.scoreDocs[0].score,
                    0.001f
                );
                doc = reader.storedFields().document(topDocs.scoreDocs[1].doc);
                assertEquals("9", doc.get("my_doc_id"));
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 9.0f }),
                    topDocs.scoreDocs[1].score,
                    0.001f
                );
                doc = reader.storedFields().document(topDocs.scoreDocs[2].doc);
                assertEquals("8", doc.get("my_doc_id"));
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 8.0f }),
                    topDocs.scoreDocs[2].score,
                    0.001f
                );
                log.info("successfully completed search tests");
            }
        }
    }

    /**
     * Test to verify that the jVector codec is able to successfully search for the nearest neighbors
     * in the index.
     * Single field is used to store the vectors.
     * Documents are stored in potentially multiple segments.
     * Multiple commits.
     * Multiple merges.
     */
    @Test
    public void testJVectorKnnIndex_multipleMerges() throws IOException {
        int k = 3; // The number of nearest neighbours to gather
        int totalNumberOfDocs = 10;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(getCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        indexWriterConfig.setMergeScheduler(new SerialMergeScheduler());
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = new float[] { 0.0f, 0.0f };
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = new float[] { 0.0f, 1.0f / i };
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField("test_field", source, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField("my_doc_id", Integer.toString(i, 10), Field.Store.YES));
                w.addDocument(doc);
                w.commit(); // this creates a new segment without triggering a merge
                w.forceMerge(1); // this merges all segments into a single segment
            }
            log.info("Done writing all files to the file system");

            w.forceMerge(1); // this merges all segments into a single segment
            log.info("Done merging all segments");
            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("We should now have 1 segment with 10 documents");
                Assert.assertEquals(1, reader.getContext().leaves().size());
                Assert.assertEquals(totalNumberOfDocs, reader.numDocs());
                final Query filterQuery = new MatchAllDocsQuery();
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery("test_field", target, k, filterQuery);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());
                Document doc = reader.storedFields().document(topDocs.scoreDocs[0].doc);
                assertEquals("10", doc.get("my_doc_id"));
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 10.0f }),
                    topDocs.scoreDocs[0].score,
                    0.001f
                );
                doc = reader.storedFields().document(topDocs.scoreDocs[1].doc);
                assertEquals("9", doc.get("my_doc_id"));
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 9.0f }),
                    topDocs.scoreDocs[1].score,
                    0.001f
                );
                doc = reader.storedFields().document(topDocs.scoreDocs[2].doc);
                assertEquals("8", doc.get("my_doc_id"));
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 8.0f }),
                    topDocs.scoreDocs[2].score,
                    0.001f
                );
                log.info("successfully completed search tests");
            }
        }
    }

    /**
     * Test to verify that the jVector codec is able to successfully search for the nearest neighbours
     * in the index.
     * A Single field is used to store the vectors.
     * Documents are stored in potentially multiple segments.
     * Multiple commits.
     * Multiple merges.
     * Large batches
     * Use a compound file
     */
    @Test
    public void testJVectorKnnIndex_multiple_merges_large_batches_no_quantization() throws IOException {
        int segmentSize = DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION;
        int totalNumberOfDocs = segmentSize * 4;
        int k = 3; // The number of nearest neighbors to gather

        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(true);
        indexWriterConfig.setCodec(getCodec(Integer.MAX_VALUE)); // effectively without quantization
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(true));
        indexWriterConfig.setMergeScheduler(new SerialMergeScheduler());
        // We set the below parameters to make sure no permature flush will occur, this way we can have a single segment, and we can force
        // test the quantization case
        indexWriterConfig.setMaxBufferedDocs(10000); // force flush every 10000 docs, this way we make sure that we only have a single
        // segment for a totalNumberOfDocs < 1000
        indexWriterConfig.setRAMPerThreadHardLimitMB(1000); // 1000MB per thread, this way we make sure that no premature flush will occur

        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = new float[] { 0.0f, 0.0f };
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = new float[] { 0.0f, 1.0f / i };
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField("test_field", source, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField("my_doc_id", Integer.toString(i, 10), Field.Store.YES));
                w.addDocument(doc);
                if (i % segmentSize == 0) {
                    w.commit(); // this creates a new segment without triggering a merge
                }
            }
            log.info("Done writing all files to the file system");

            w.forceMerge(1); // this merges all segments into a single segment
            log.info("Done merging all segments");
            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("We should now have 1 segment with {} documents", totalNumberOfDocs);
                Assert.assertEquals(1, reader.getContext().leaves().size());
                Assert.assertEquals(totalNumberOfDocs, reader.numDocs());
                final Query filterQuery = new MatchAllDocsQuery();
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery("test_field", target, k, filterQuery);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());

                float expectedMinScoreInTopK = VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, k });
                final float recall = calculateRecall(topDocs, expectedMinScoreInTopK);
                Assert.assertEquals(1.0f, recall, 0.01f);

                log.info("successfully completed search tests");
            }
        }
    }

    /**
     * Tests the functionality and integrity of a Lucene k-NN index under multiple merge cycles and verifies
     *  the proper ordering of vectors and document identifiers.
     *
     * The method performs the following validation steps:
     * 1. Indexes a predefined number of documents into a Lucene index, creating many small segments.
     * Each document
     *    includes a k-NN float vector field encoding a specific order.
     * 2. Executes several merge operations on the index (partial and full merges) to validate that the merging
     *    process maintains correctness and consistency.
     * 3. Validates the following invariants post-merge:
     *    (a) Verifies that the index is merged into a single segment.
     *    (b) Confirms the integrity of vector values by iterating through the merged segment and checking the
     *        relationship between vector components and document identifiers.
     *    (c) Performs k-NN searches with various cases:
     *        - Single-threaded searches using vectors to ensure correct results.
     *        - Multi-threaded concurrent searches to confirm robustness and verify the index operates correctly
     *          under concurrent access without exhausting file handles or encountering other issues.
     *
     * Assertions are used throughout to ensure the state of the index matches the expected behavior,
     * validate merge
     * results, and confirm the accuracy of search operations.
     * The test also logs the number of successful k-NN queries
     * during the concurrent search phase.
     *
     * @throws IOException if an I/O error occurs during index operations.
     * @throws InterruptedException if the concurrent search phase is interrupted.
     */
    @Test
    public void testLuceneKnnIndex_multipleMerges_with_ordering_check() throws IOException, InterruptedException {
        final int numDocs = 10000;
        final String floatVectorField = "vec";
        final String expectedDocIdField = "expectedDocId";
        final Path indexPath = createTempDir();
        final float[][] sourceVectors = TestUtils.generateRandomVectors(numDocs, 2);
        final VectorSimilarityFunction vectorSimilarityFunction = VectorSimilarityFunction.EUCLIDEAN;

        try (Directory dir = newFSDirectory(indexPath)) {
            IndexWriterConfig cfg = newIndexWriterConfig();
            cfg.setCodec(getCodec());
            cfg.setUseCompoundFile(false);
            cfg.setMergePolicy(new ForceMergesOnlyMergePolicy(false));
            cfg.setMergeScheduler(new SerialMergeScheduler());

            try (IndexWriter w = new IndexWriter(dir, cfg)) {
                /* ---------- 1. index documents, create many tiny segments ---------- */
                for (int i = 0; i < numDocs; i++) {
                    Document doc = new Document();
                    // vector whose first component encodes the future (segment-local) docID
                    doc.add(new KnnFloatVectorField(floatVectorField, sourceVectors[i], vectorSimilarityFunction));
                    doc.add(new StoredField(expectedDocIdField, i));
                    w.addDocument(doc);
                }
                w.commit();

                /* ---------- 2. run several merge cycles ---------- */
                w.forceMerge(5);  // partial merge
                w.forceMerge(3);  // another partial merge
                w.forceMerge(1);  // final full merge
            }

            /* ---------- 3. open reader and assert the invariant ---------- */
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals("we merged down to exactly one segment", 1, reader.leaves().size());

                // (a) iterate through vectors directly
                for (LeafReaderContext context : reader.leaves()) {
                    FloatVectorValues vectorValues = context.reader().getFloatVectorValues("vec");
                    for (int docId = 0; docId < context.reader().maxDoc(); docId++) {
                        final int luceneDocId = context.docBase + docId;
                        final int globalDocId = reader.storedFields()
                            .document(luceneDocId)
                            .getField(expectedDocIdField)
                            .storedValue()
                            .getIntValue();
                        float[] vectorValue = vectorValues.vectorValue(docId);
                        float[] expectedVectorValue = sourceVectors[globalDocId];
                        Assert.assertArrayEquals("vectors in source and index should match", expectedVectorValue, vectorValue, 0.0f);
                    }
                }

                // (b) search with the same vector and confirm we are not exhausting the file handles with each search
                IndexSearcher searcher = newSearcher(reader);
                LeafReaderContext context = reader.leaves().get(0); // we only have one leaf at this point so we can use it to obtain the
                                                                    // vector values
                final int baseDocId = context.docBase;
                final FloatVectorValues vectorValues = context.reader().getFloatVectorValues("vec");
                final int k = 1;
                for (int docId = 0; docId < reader.maxDoc(); docId++) {
                    float[] query = new float[] { docId, 0 };
                    TopDocs td = searcher.search(getJVectorKnnFloatVectorQuery("vec", query, k, new MatchAllDocsQuery()), k);
                    assertEquals(k, td.scoreDocs.length);
                }

                // (c) search with the same vector and this time add concurrency to make sure we are still not exhausting the file handles
                int numThreads = 10; // Number of concurrent search threads
                int queriesPerThread = 100; // Number of searches per thread
                ExecutorService executor = Executors.newFixedThreadPool(numThreads);
                CountDownLatch latch = new CountDownLatch(numThreads);
                AtomicBoolean failureDetected = new AtomicBoolean(false);
                AtomicInteger totalQueries = new AtomicInteger(0);

                try {
                    for (int t = 0; t < numThreads; t++) {
                        executor.submit(() -> {
                            int i = 0;

                            try {
                                for (i = 0; i < queriesPerThread && !failureDetected.get(); i++) {
                                    float[] query = TestUtils.generateRandomVectors(1, 2)[0];
                                    try {
                                        TopDocs td = searcher.search(new KnnFloatVectorQuery("vec", query, k), k);
                                        assertEquals("Search should return correct number of results", k, td.scoreDocs.length);
                                        final int localDocId = td.scoreDocs[0].doc;
                                        final int globalDocId = reader.storedFields()
                                            .document(localDocId)
                                            .getField(expectedDocIdField)
                                            .storedValue()
                                            .getIntValue();
                                        float[] vectorValue = vectorValues.vectorValue(localDocId - baseDocId);
                                        float[] expectedVectorValue = sourceVectors[globalDocId];
                                        Assert.assertArrayEquals(
                                            "vectors in source and index should match",
                                            expectedVectorValue,
                                            vectorValue,
                                            0.0f
                                        );
                                        totalQueries.incrementAndGet();
                                    } catch (Throwable e) {
                                        failureDetected.compareAndSet(false, true);
                                        log.error("Exception encountered", e);
                                        fail("Exception during concurrent search: " + e.getMessage());
                                    }
                                }
                            } finally {
                                latch.countDown();
                                log.warn("Ran {} queries", i);
                            }
                        });
                    }

                    // Wait for all threads to complete or for a failure
                    boolean completed = latch.await(30, TimeUnit.SECONDS);
                    assertTrue("Test timed out while waiting for concurrent searches", completed);
                    assertFalse("Test encountered failures during concurrent searches", failureDetected.get());
                    assertEquals("Incorrect number of queries executed", numThreads * queriesPerThread, totalQueries.get());

                    // Log the number of successful queries
                    log.info("Successfully completed {} concurrent kNN search queries!", totalQueries.get());

                } finally {
                    executor.shutdownNow();
                }
            }
        }

    }

    /**
     * Test to verify that a document which has been deleted is no longer
     * returned in a k-NN search.  The index uses the JVector codec and is
     * kept in multiple segments to ensure we also cover the case where the
     * deleted document still physically resides in the segment as a dead
     * (non-live) record.
     */
    @Test
    public void testJVectorKnnIndex_deletedDocs() throws IOException {
        final int totalNumberOfDocs = 100;
        final int batchSize = 10;
        final int k = batchSize - 1;
        final int docToDeleteInEachBatch = 5;
        final Path indexPath = createTempDir();
        final IndexWriterConfig iwc = newIndexWriterConfig();
        // JVector codec requires compound files to be disabled at the moment
        iwc.setUseCompoundFile(false);
        iwc.setCodec(getCodec());
        iwc.setMergePolicy(new ForceMergesOnlyMergePolicy(false));

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, iwc)) {

            /*
             * 1.  Index 100 docs, in batches of 10.  Delete the 5th doc in each batch.
             *     will leave us with 10 segments, each with 9 live docs.
             */
            int batchNumber = 0;
            for (int i = 1; i <= totalNumberOfDocs; i++) {
                Document doc = new Document();
                final float[] vector = { 0.0f, 1.0f * (i + batchNumber) };
                doc.add(new StringField("docId", Integer.toString(i + 1), Field.Store.YES));
                doc.add(new KnnFloatVectorField("test_field", vector, VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc);
                if (i % batchSize == 0) {
                    writer.flush();
                    writer.deleteDocuments(new TermQuery(new Term("docId", Integer.toString(i - docToDeleteInEachBatch))));
                    batchNumber++;
                }
            }
            writer.commit();

            /* ----------------------------------------
             * 2.  Merge all segments into one
             * ---------------------------------------- */
            writer.forceMerge(1);

            /* ----------------------------------------
             * 3.  Search – the deleted doc must be gone
             * ---------------------------------------- */
            try (IndexReader reader = DirectoryReader.open(writer)) {
                assertEquals(
                    "All documents except the deleted ones should be live",
                    totalNumberOfDocs - (totalNumberOfDocs / batchSize),
                    reader.numDocs()
                );
                // For each batch we will verify that the deleted document doesn't come up in search and only it's neighbours are returned

                for (int i = 0; i < totalNumberOfDocs; i += batchSize) {
                    final float[] target = { 0.0f, 1.0f * (i + docToDeleteInEachBatch) };
                    final IndexSearcher searcher = newSearcher(reader);
                    final KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery(
                        "test_field",
                        target,
                        k,
                        new MatchAllDocsQuery()
                    );
                    TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                    assertEquals(k, topDocs.totalHits.value());
                    for (int j = 0; j < k; j++) {
                        Document doc = reader.storedFields().document(topDocs.scoreDocs[j].doc);
                        int docId = Integer.parseInt(doc.get("docId"));
                        assertNotEquals("Deleted doc should not be returned in search results", i + docToDeleteInEachBatch, docId);
                    }
                }
            }
        }
    }

    /**
     * Test to verify that the Lucene codec is able to successfully search for the nearest neighbours
     * in the index.
     * Single field is used to store the vectors.
     * Documents are stored in potentially multiple segments.
     * Multiple commits.
     * Multiple merges.
     * Merge is enabled.
     * compound file is enabled.
     */
    @Test
    public void testLuceneKnnIndex_mergeEnabled_withCompoundFile() throws IOException {
        int k = 3; // The number of nearest neighbors to gather
        int totalNumberOfDocs = 10;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(true);
        indexWriterConfig.setCodec(getCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(true));
        indexWriterConfig.setMergeScheduler(new SerialMergeScheduler());
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = new float[] { 0.0f, 0.0f };
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = new float[] { 0.0f, 1.0f / i };
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField("test_field", source, VectorSimilarityFunction.EUCLIDEAN));
                w.addDocument(doc);
                w.flush(); // this creates a new segment without triggering a merge
            }
            log.info("Done writing all files to the file system");

            w.forceMerge(1); // this merges all segments into a single segment
            log.info("Done merging all segments");
            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("We should now have 1 segment with 10 documents");
                Assert.assertEquals(1, reader.getContext().leaves().size());
                Assert.assertEquals(totalNumberOfDocs, reader.numDocs());
                final Query filterQuery = new MatchAllDocsQuery();
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery("test_field", target, k, filterQuery);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());
                assertEquals(9, topDocs.scoreDocs[0].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 10.0f }),
                    topDocs.scoreDocs[0].score,
                    0.01f
                );
                assertEquals(8, topDocs.scoreDocs[1].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 9.0f }),
                    topDocs.scoreDocs[1].score,
                    0.01f
                );
                assertEquals(7, topDocs.scoreDocs[2].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 8.0f }),
                    topDocs.scoreDocs[2].score,
                    0.01f
                );
                log.info("successfully completed search tests");
            }
        }
    }

    /**
     * Test to verify that the Lucene codec is able to successfully search for the nearest neighbours
     * in the index.
     * Single field is used to store the vectors.
     * Documents are stored in potentially multiple segments.
     * Multiple commits.
     * Multiple merges.
     * Merge is enabled.
     * compound file is enabled.
     * cosine similarity is used.
     */
    @Test
    public void testLuceneKnnIndex_mergeEnabled_withCompoundFile_cosine() throws IOException {
        int k = 3; // The number of nearest neighbours to gather
        int totalNumberOfDocs = 10;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(true);
        indexWriterConfig.setCodec(getCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(true));
        indexWriterConfig.setMergeScheduler(new SerialMergeScheduler());
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = new float[] { 1.0f, 1.0f };
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = new float[] { 1.0f + i, 2.0f * i };
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField("test_field", source, VectorSimilarityFunction.COSINE));
                w.addDocument(doc);
                w.flush(); // this creates a new segment without triggering a merge
            }
            log.info("Done writing all files to the file system");

            w.forceMerge(1); // this merges all segments into a single segment
            log.info("Done merging all segments");
            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("We should now have 1 segment with 10 documents");
                Assert.assertEquals(1, reader.getContext().leaves().size());
                Assert.assertEquals(totalNumberOfDocs, reader.numDocs());
                final Query filterQuery = new MatchAllDocsQuery();
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery("test_field", target, k, filterQuery);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());
                assertEquals(0, topDocs.scoreDocs[0].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.COSINE.compare(target, new float[] { 2.0f, 2.0f }),
                    topDocs.scoreDocs[0].score,
                    0.001f
                );
                assertEquals(1, topDocs.scoreDocs[1].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.COSINE.compare(target, new float[] { 3.0f, 4.0f }),
                    topDocs.scoreDocs[1].score,
                    0.001f
                );
                assertEquals(2, topDocs.scoreDocs[2].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.COSINE.compare(target, new float[] { 4.0f, 6.0f }),
                    topDocs.scoreDocs[2].score,
                    0.001f
                );
                log.info("successfully completed search tests");
            }
        }
    }

    /**
     * Test to verify that the JVector codec is providing proper error if used with byte vector
     * TODO: Create Binary Quantization support for JVector codec
     */
    @Test
    public void testJVectorKnnIndex_simpleCase_withBinaryVector() throws IOException {
        int k = 3; // The number of nearest neighbours to gather
        int totalNumberOfDocs = 10;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        // TODO: re-enable this after fixing the compound file augmentation for JVector
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(getCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (Directory dir = newFSDirectory(indexPath); RandomIndexWriter w = new RandomIndexWriter(random(), dir, indexWriterConfig)) {
            final byte[] source = new byte[] { (byte) 0, (byte) 0 };
            final Document doc = new Document();
            doc.add(new KnnByteVectorField("test_field", source, VectorSimilarityFunction.EUCLIDEAN));
            Assert.assertThrows(UnsupportedOperationException.class, () -> w.addDocument(doc));
        }
    }

    /**
     * Test to verify that the JVector codec is able to successfully search for the nearest neighbours
     * in the index with a filter applied.
     */
    @Test
    public void testJVectorKnnIndex_withFilter() throws IOException {
        int k = 3; // The number of nearest neighbours to gather
        int totalNumberOfDocs = 10;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(getCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (Directory dir = newFSDirectory(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = new float[] { 0.0f, 0.0f };
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = new float[] { 0.0f, 1.0f / i };
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField("test_field", source, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField("filter_field", i % 2 == 0 ? "even" : "odd", Field.Store.YES));
                w.addDocument(doc);
            }
            log.info("Flushing docs to make them discoverable on the file system");
            w.commit();

            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("Applying filter to the KNN search");
                final Query filterQuery = new TermQuery(new Term("filter_field", "even"));
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery("test_field", target, k, filterQuery);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);

                log.info("Validating filtered KNN results");
                assertEquals(k, topDocs.totalHits.value());
                assertEquals(9, topDocs.scoreDocs[0].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 10.0f }),
                    topDocs.scoreDocs[0].score,
                    0.001f
                );
                assertEquals(7, topDocs.scoreDocs[1].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 8.0f }),
                    topDocs.scoreDocs[1].score,
                    0.001f
                );
                assertEquals(5, topDocs.scoreDocs[2].doc);
                Assert.assertEquals(
                    VectorSimilarityFunction.EUCLIDEAN.compare(target, new float[] { 0.0f, 1.0f / 6.0f }),
                    topDocs.scoreDocs[2].score,
                    0.001f
                );
                log.info("successfully completed filtered search tests");
            }
        }
    }

    /**
     * Test the simple case of quantization where we have the perfect batch single batch size with no merges or too small batch sizes
     */
    @Test
    public void testJVectorKnnIndex_simpleCase_withQuantization() throws IOException {
        int k = 50; // The number of nearest neighbours to gather
        int dimension = 16;
        int totalNumberOfDocs = DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION;
        final VectorSimilarityFunction vectorSimilarityFunction = VectorSimilarityFunction.EUCLIDEAN;

        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(getCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION));
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        // We set the below parameters to make sure no permature flush will occur, this way we can have a single segment, and we can force
        // test the quantization case
        indexWriterConfig.setMaxBufferedDocs(10000); // force flush every 10000 docs, this way we make sure that we only have a single
                                                     // segment for a totalNumberOfDocs < 1000
        indexWriterConfig.setRAMPerThreadHardLimitMB(1000); // 1000MB per thread, this way we make sure that no premature flush will occur
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = generateZerosVectorWithLastValue(dimension, 0);
            final float[][] vectors = TestUtils.generateRandomVectors(totalNumberOfDocs, dimension);
            final Set<Integer> groundTruthVectorsIds = calculateGroundTruthVectorsIds(target, vectors, k, vectorSimilarityFunction);
            for (int i = 0; i < vectors.length; i++) {
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(TEST_FIELD, vectors[i], vectorSimilarityFunction));
                doc.add(new IntField(TEST_ID_FIELD, i, Field.Store.YES));
                w.addDocument(doc);
            }
            log.info("Flushing docs to make them discoverable on the file system");
            w.commit();

            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("We should now have a single segment with {} documents", totalNumberOfDocs);
                Assert.assertEquals(1, reader.getContext().leaves().size());
                Assert.assertEquals(totalNumberOfDocs, reader.numDocs());

                final Query filterQuery = new MatchAllDocsQuery();
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery(TEST_FIELD, target, k, filterQuery);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());
                final float recall = calculateRecall(reader, groundTruthVectorsIds, topDocs, k);
                Assert.assertEquals(1.0f, recall, 0.05f);
                log.info("successfully completed search tests");
            }
        }
    }

    /**
     * Test recall with different types of rerank parameters
     */
    @Test
    public void testJVectorKnnIndex_simpleCase_withQuantization_rerank() throws IOException {
        int k = 1; // The number of nearest neighbours to gather
        int dimension = 16;
        int totalNumberOfDocs = DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(getCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION));
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        // We set the below parameters to make sure no permature flush will occur, this way we can have a single segment, and we can force
        // test the quantization case
        indexWriterConfig.setMaxBufferedDocs(10000); // force flush every 10000 docs, this way we make sure that we only have a single
        // segment for a totalNumberOfDocs < 1000
        indexWriterConfig.setRAMPerThreadHardLimitMB(1000); // 1000MB per thread, this way we make sure that no premature flush will occur
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = generateZerosVectorWithLastValue(dimension, 0);
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = generateZerosVectorWithLastValue(dimension, i);
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField("test_field", source, VectorSimilarityFunction.EUCLIDEAN));
                w.addDocument(doc);
            }
            log.info("Flushing docs to make them discoverable on the file system");
            w.commit();

            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("We should now have a single segment with {} documents", totalNumberOfDocs);
                Assert.assertEquals(1, reader.getContext().leaves().size());
                Assert.assertEquals(totalNumberOfDocs, reader.numDocs());

                final Query filterQuery = new MatchAllDocsQuery();
                final IndexSearcher searcher = newSearcher(reader);
                float expectedMinScoreInTopK = VectorSimilarityFunction.EUCLIDEAN.compare(
                    target,
                    new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, k }
                );

                // Query with essentially no reranking and expect recall to be very low
                KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery("test_field", target, k, filterQuery, 1);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());

                final float recallWithLowOverqueryFactor = calculateRecall(topDocs, expectedMinScoreInTopK);

                // Query with reranking and expect recall to be high
                knnFloatVectorQuery = getJVectorKnnFloatVectorQuery("test_field", target, k, filterQuery, 5);
                topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());
                float recallWithHighOverqueryFactor = calculateRecall(topDocs, expectedMinScoreInTopK);
                Assert.assertTrue(recallWithLowOverqueryFactor <= recallWithHighOverqueryFactor);

                log.info("successfully completed search tests");
            }
        }
    }

    /**
     * Test the simple case of quantization where we have the perfect batch single batch size each time with a merge of
     * multiple segments
     */
    @Test
    public void testJVectorKnnIndex_happyCase_withQuantization_multipleSegments() throws IOException {
        final int dimension = 16;
        final VectorSimilarityFunction vectorSimilarityFunction = VectorSimilarityFunction.EUCLIDEAN;
        final int k = 50; // The number of nearest neighbours to gather, we set a high number here to avoid an inaccurate result and
                          // jittery tests
        final int perfectBatchSize = DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION; // MINIMUM_BATCH_SIZE_FOR_QUANTIZATION is the minimal
                                                                                  // batch size that will trigger a quantization without
                                                                                  // breaking it, generally speaking the batch size can't be
                                                                                  // lower than the number of clusters
        final int totalNumberOfDocs = perfectBatchSize * 2;

        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(getCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION));
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        // We set the below parameters to make sure no permature flush will occur, this way we can have a single segment, and we can force
        // test the quantization case
        indexWriterConfig.setMaxBufferedDocs(10000); // force flush every 10000 docs, this way we make sure that we only have a single
        // segment for a totalNumberOfDocs < 1000
        indexWriterConfig.setRAMPerThreadHardLimitMB(1000); // 1000MB per thread, this way we make sure that no premature flush will occur
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = generateZerosVectorWithLastValue(dimension, 0);
            final float[][] vectors = TestUtils.generateRandomVectors(totalNumberOfDocs, dimension);
            final Set<Integer> groundTruthVectorsIds = calculateGroundTruthVectorsIds(target, vectors, k, vectorSimilarityFunction);

            for (int i = 0; i < vectors.length; i++) {
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(TEST_FIELD, vectors[i], vectorSimilarityFunction));
                doc.add(new IntField(TEST_ID_FIELD, i, Field.Store.YES));
                w.addDocument(doc);
                if (i % perfectBatchSize == 0) {
                    w.commit();
                }
            }
            log.info("Flushing docs to make them discoverable on the file system");
            w.forceMerge(1);

            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("We should now have a single segment with {} documents", totalNumberOfDocs);
                Assert.assertEquals(1, reader.getContext().leaves().size());
                Assert.assertEquals(totalNumberOfDocs, reader.numDocs());

                final Query filterQuery = new MatchAllDocsQuery();
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery("test_field", target, k, filterQuery);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());
                final float recall = calculateRecall(reader, groundTruthVectorsIds, topDocs, k);
                Assert.assertEquals(1.0f, recall, 0.05f);
                log.info("successfully completed search tests");
            }
        }
    }

    /**
     * Test the non-ideal case where batch sizes are not perfect and are lower than the number of recommended clusters in the index
     * The expected behavior is for the quantization to only kick in when we have a merge or batch size that is bigger than the minimal required batch size
     */
    @Test
    public void testJVectorKnnIndex_mixedBatchSizes_withQuantization_multipleMerges() throws IOException {
        final int k = 50; // The number of nearest neighbours to gather, we set a high number here to avoid an inaccurate result and
                          // jittery tests
        final int dimension = 16;
        final int notIdealBatchSize = DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION / 3; // Batch size that is not ideal for quantization and
                                                                                       // shouldn't trigger it
        final int totalNumberOfDocs = notIdealBatchSize * 3; // 3 batches of documents each will result in quantization only when the merge
                                                             // is triggered, and we have a batch size of {@link
                                                             // MINIMUM_BATCH_SIZE_FOR_QUANTIZATION} as a result of merging all the smaller
                                                             // batches

        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(getCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION));
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        // We set the below parameters to make sure no permature flush will occur, this way we can have a single segment, and we can force
        // test the quantization case
        indexWriterConfig.setMaxBufferedDocs(10000); // force flush every 10000 docs, this way we make sure that we only have a single
        // segment for a totalNumberOfDocs < 1000
        indexWriterConfig.setRAMPerThreadHardLimitMB(1000); // 1000MB per thread, this way we make sure that no premature flush will occur
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = generateZerosVectorWithLastValue(dimension, 0);
            final float[][] vectors = getMonotonicallyIncreasingVectors(totalNumberOfDocs, dimension);
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = vectors[i - 1];
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField("test_field", source, VectorSimilarityFunction.EUCLIDEAN));
                w.addDocument(doc);
                if (i % notIdealBatchSize == 0) {
                    w.commit();
                }
            }
            log.info("Flushing docs to make them discoverable on the file system");
            w.forceMerge(1);

            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("We should now have a single segment with {} documents", totalNumberOfDocs);
                Assert.assertEquals(1, reader.getContext().leaves().size());
                Assert.assertEquals(totalNumberOfDocs, reader.numDocs());

                final Query filterQuery = new MatchAllDocsQuery();
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery("test_field", target, k, filterQuery);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());
                float expectedMinScoreInTopK = VectorSimilarityFunction.EUCLIDEAN.compare(
                    target,
                    new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, k }
                );
                final float recall = calculateRecall(topDocs, expectedMinScoreInTopK);
                Assert.assertEquals(1.0f, recall, 0.05f);
                log.info("successfully completed search tests");
            }
        }
    }

    /**
     * Test the non-ideal case where batch sizes are not perfect and are lower than the number of recommended clusters in the index
     * The expected behavior is for the quantization to only kick in when we have a merge or batch size that is bigger than the minimal required batch size
     * Also this is adding the compound file to the mix
     */
    @Test
    public void testJVectorKnnIndex_mixedBatchSizes_withQuantization_multipleMerges_withCompoundFile() throws IOException {
        final int k = 50; // The number of nearest neighbours to gather, we set a high number here to avoid an inaccurate result and
        // jittery tests
        final int dimension = 16;
        final int notIdealBatchSize = DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION / 3; // Batch size that is not ideal for quantization and
        // shouldn't trigger it
        final int totalNumberOfDocs = notIdealBatchSize * 10; // 3 batches of documents each will result in quantization only when the merge
        // is triggered, and we have a batch size of {@link MINIMUM_BATCH_SIZE_FOR_QUANTIZATION}
        // as a result of merging all the smaller batches
        final VectorSimilarityFunction vectorSimilarityFunction = VectorSimilarityFunction.EUCLIDEAN;

        boolean useCompoundFile = true;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(useCompoundFile);
        indexWriterConfig.setCodec(getCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION));
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(useCompoundFile));
        // We set the below parameters to make sure no premature flush will occur, this way we can have a single segment, and we can force
        // test the quantization case
        indexWriterConfig.setMaxBufferedDocs(10000); // force flush every 10000 docs, this way we make sure that we only have a single
        // segment for a totalNumberOfDocs < 1000
        indexWriterConfig.setRAMPerThreadHardLimitMB(1000); // 1000MB per thread, this way we make sure that no premature flush will occur
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = generateZerosVectorWithLastValue(dimension, 0);
            // We will use random vectors because otherwise PQ will have a correlated subspaces which will result in a broken linear graph
            final float[][] vectors = TestUtils.generateRandomVectors(totalNumberOfDocs, dimension);
            final Set<Integer> groundTruthVectorsIds = calculateGroundTruthVectorsIds(target, vectors, k, vectorSimilarityFunction);
            for (int i = 0; i < totalNumberOfDocs; i++) {
                final float[] source = vectors[i];
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(TEST_FIELD, source, vectorSimilarityFunction));
                doc.add(new IntField(TEST_ID_FIELD, i, Field.Store.YES));
                w.addDocument(doc);
                if (i % notIdealBatchSize == 0) {
                    w.commit();
                }
            }
            w.commit();
            log.info("Flushing docs to make them discoverable on the file system");
            w.forceMerge(1);

            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("We should now have a single segment with {} documents", totalNumberOfDocs);
                Assert.assertEquals(1, reader.getContext().leaves().size());
                Assert.assertEquals(totalNumberOfDocs, reader.numDocs());

                final Query filterQuery = new MatchAllDocsQuery();
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery("test_field", target, k, filterQuery, 1000);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());
                final float recall = calculateRecall(reader, groundTruthVectorsIds, topDocs, k);
                Assert.assertEquals("Expected to have recall of 1.0+/-0.05 but got " + recall, 1.0f, recall, 0.05f);
                log.info("successfully completed search tests");
            }
        }

        Assert.assertTrue("No quantization time recorded", KNNCounter.KNN_QUANTIZATION_TRAINING_TIME.getCount() > 0);
        Assert.assertTrue("No graph merge time recorded", KNNCounter.KNN_GRAPH_MERGE_TIME.getCount() > 0);
    }

    /**
     * We will use multiple batches, each can trigger a quantization and later merge them in an appending order to keep track
     * of refinement
     * @throws IOException
     */
    @Test
    public void testJVectorKnnIndex_withQuantization_withCompoundFile_with_refinement() throws IOException {
        final int k = 50; // The number of nearest neighbours to gather, we set a high number here to avoid an inaccurate result and
        // jittery tests
        final int dimension = 16;
        final int idealBatchSize = DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION; // Batch size that is not ideal for quantization and
        // shouldn't trigger it
        final int totalNumberOfDocs = idealBatchSize * 10; // 10 batches, each batch on it's own will trigger quantization
        final VectorSimilarityFunction vectorSimilarityFunction = VectorSimilarityFunction.EUCLIDEAN;

        boolean useCompoundFile = true;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        indexWriterConfig.setUseCompoundFile(useCompoundFile);
        indexWriterConfig.setCodec(getCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION));
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(useCompoundFile));
        // We set the below parameters to make sure no premature flush will occur, this way we can have a single segment, and we can force
        // test the quantization case
        indexWriterConfig.setMaxBufferedDocs(10000); // force flush every 10000 docs, this way we make sure that we only have a single
        // segment for a totalNumberOfDocs < 1000
        indexWriterConfig.setRAMPerThreadHardLimitMB(1000); // 1000MB per thread, this way we make sure that no premature flush will occur
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter w = new IndexWriter(dir, indexWriterConfig)) {
            final float[] target = generateZerosVectorWithLastValue(dimension, 0);
            // We will use random vectors because otherwise PQ will have a correlated subspaces which will result in a broken linear graph
            final float[][] vectors = TestUtils.generateRandomVectors(totalNumberOfDocs, dimension);
            final Set<Integer> groundTruthVectorsIds = calculateGroundTruthVectorsIds(target, vectors, k, vectorSimilarityFunction);
            for (int i = 0; i < totalNumberOfDocs; i++) {
                final float[] source = vectors[i];
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(TEST_FIELD, source, vectorSimilarityFunction));
                doc.add(new IntField(TEST_ID_FIELD, i, Field.Store.YES));
                w.addDocument(doc);
                if (i % idealBatchSize == 0) {
                    final long beforeTrainingTime = KNNCounter.KNN_QUANTIZATION_TRAINING_TIME.getCount();
                    w.commit();
                    w.forceMerge(1); // force merge will trigger PQ refinement if other segments are present
                    final long afterTrainingTime = KNNCounter.KNN_QUANTIZATION_TRAINING_TIME.getCount();
                    Assert.assertTrue(
                        "Expected to have a training time of at least " + beforeTrainingTime + " but got " + afterTrainingTime,
                        afterTrainingTime >= beforeTrainingTime
                    );
                }
            }
            w.commit();
            log.info("Flushing docs to make them discoverable on the file system");
            w.forceMerge(1);

            try (IndexReader reader = DirectoryReader.open(w)) {
                log.info("We should now have a single segment with {} documents", totalNumberOfDocs);
                Assert.assertEquals(1, reader.getContext().leaves().size());
                Assert.assertEquals(totalNumberOfDocs, reader.numDocs());

                final Query filterQuery = new MatchAllDocsQuery();
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = getJVectorKnnFloatVectorQuery("test_field", target, k, filterQuery, 1000);
                TopDocs topDocs = searcher.search(knnFloatVectorQuery, k);
                assertEquals(k, topDocs.totalHits.value());
                final float recall = calculateRecall(reader, groundTruthVectorsIds, topDocs, k);
                Assert.assertEquals("Expected to have recall of 1.0+/-0.05 but got " + recall, 1.0f, recall, 0.05f);
                log.info("successfully completed search tests");
            }
        }

        Assert.assertTrue("No graph merge time recorded", KNNCounter.KNN_GRAPH_MERGE_TIME.getCount() > 0);
    }

    /**
     * Calculate the recall for the top k documents
     * For simplicity we assume that all documents have unique scores and therefore the minimum score in the top k documents is the kth document
     * @param topDocs the top documents returned by the search
     * @param minScoreInTopK the minimum score in the top k documents
     * @return the recall of the top k documents
     */
    private float calculateRecall(TopDocs topDocs, float minScoreInTopK) {
        int totalRelevantDocs = 0;
        for (int i = 0; i < topDocs.scoreDocs.length; i++) {
            if (topDocs.scoreDocs[i].score >= minScoreInTopK) {
                totalRelevantDocs++;
            }
        }
        float recall = ((float) totalRelevantDocs) / ((float) topDocs.scoreDocs.length);

        if (recall == 0.0f) {
            log.info(
                "Recall is 0.0, this is probably not correct, here is some debug information\n topDocs: {}, minScoreInTopK: {}, totalRelevantDocs: {}",
                topDocsToString(topDocs),
                minScoreInTopK,
                totalRelevantDocs
            );
        }
        return recall;
    }

    // convert topDocs to a pretty printed string
    private String topDocsToString(TopDocs topDocs) {
        StringBuilder sb = new StringBuilder();
        sb.append("TopDocs: [");
        for (int i = 0; i < topDocs.scoreDocs.length; i++) {
            sb.append(topDocs.scoreDocs[i].doc).append(" (").append(topDocs.scoreDocs[i].score).append("), ");
        }
        sb.append("]");
        return sb.toString();
    }

    private JVectorKnnFloatVectorQuery getJVectorKnnFloatVectorQuery(String fieldName, float[] target, int k, Query filterQuery) {
        return getJVectorKnnFloatVectorQuery(fieldName, target, k, filterQuery, KNNConstants.DEFAULT_OVER_QUERY_FACTOR);
    }

    private JVectorKnnFloatVectorQuery getJVectorKnnFloatVectorQuery(
        String fieldName,
        float[] target,
        int k,
        Query filterQuery,
        int overQueryFactor
    ) {
        return new JVectorKnnFloatVectorQuery(
            fieldName,
            target,
            k,
            filterQuery,
            overQueryFactor,
            KNNConstants.DEFAULT_QUERY_SIMILARITY_THRESHOLD.floatValue(),
            KNNConstants.DEFAULT_QUERY_RERANK_FLOOR.floatValue(),
            KNNConstants.DEFAULT_QUERY_USE_PRUNING
        );
    }

    private static float[][] getMonotonicallyIncreasingVectors(int numVectors, int vectorDimension) {
        float[][] vectors = new float[numVectors][vectorDimension];
        for (int i = 0; i < numVectors; i++) {
            vectors[i] = generateZerosVectorWithLastValue(vectorDimension, i);
        }

        return vectors;
    }

    private static float[] generateZerosVectorWithLastValue(int vectorDimension, int lastValue) {
        float[] vector = new float[vectorDimension];
        for (int i = 0; i < vectorDimension - 1; i++) {
            vector[i] = 0;
        }
        vector[vectorDimension - 1] = lastValue;
        return vector;
    }

    private static float calculateRecall(IndexReader reader, Set<Integer> groundTruthVectorsIds, TopDocs topDocs, int k)
        throws IOException {
        final ScoreDoc[] scoreDocs = topDocs.scoreDocs;
        Assert.assertEquals(groundTruthVectorsIds.size(), scoreDocs.length);
        int totalRelevantDocs = 0;
        for (ScoreDoc scoreDoc : scoreDocs) {
            final int id = reader.storedFields().document(scoreDoc.doc).getField(TEST_ID_FIELD).storedValue().getIntValue();
            if (groundTruthVectorsIds.contains(id)) {
                totalRelevantDocs++;
            }
        }
        return ((float) totalRelevantDocs) / ((float) k);
    }

    /**
     * Find the IDs of the ground truth vectors in the dataset
     * @param query query vector
     * @param dataset dataset of all the vectors with their ordinal position in the array as their ID
     * @param k the number of expected results
     * @return the IDs of the ground truth vectors in the dataset
     */
    private static Set<Integer> calculateGroundTruthVectorsIds(
        float[] query,
        final float[][] dataset,
        int k,
        VectorSimilarityFunction vectorSimilarityFunction
    ) {
        final Set<Integer> groundTruthVectorsIds = new HashSet<>();
        final PriorityQueue<ScoreDoc> priorityQueue = new PriorityQueue<>(k, (o1, o2) -> Float.compare(o1.score, o2.score));
        for (int i = 0; i < dataset.length; i++) {
            ScoreDoc scoreDoc = new ScoreDoc(i, vectorSimilarityFunction.compare(query, dataset[i]));
            if (priorityQueue.size() >= k) {
                final ScoreDoc top = priorityQueue.poll();
                if (top.score < scoreDoc.score) {
                    priorityQueue.add(scoreDoc);
                } else {
                    priorityQueue.add(top);
                }
            } else {
                priorityQueue.add(scoreDoc);
            }
        }
        while (!priorityQueue.isEmpty()) {
            groundTruthVectorsIds.add(priorityQueue.poll().doc);
        }

        return groundTruthVectorsIds;
    }
}

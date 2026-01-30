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
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Assert;
import org.junit.Test;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;

import java.io.IOException;
import java.nio.file.Path;

import static org.opensearch.knn.index.engine.CommonTestUtils.getCodec;

/**
 * Comprehensive test class for JVectorWriter merge scenarios with deletions.
 * 
 * This test suite specifically validates the merge behavior described in JVectorWriter.merge():
 * 
 * 1. **No PQ, No Deletes**: Incremental graph building by expanding the leading segment's graph
 * 2. **No PQ, With Deletes**: Incremental graph building with cleanup of deleted nodes
 * 3. **With PQ (new codebooks)**: Build graph from scratch using PQ vectors
 * 4. **With PQ (refined codebooks)**: Refine existing PQ codebooks and build graph from scratch
 * 
 * The tests cover multiple rounds of merges with deletions to ensure the merge logic
 * handles complex scenarios correctly.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "")
@Log4j2
public class JVectorWriterDeleteMergeTests extends LuceneTestCase {

    private static final String TEST_FIELD = "test_field";
    private static final String DOC_ID_FIELD = "docId";

    /**
     * Test Scenario 1: Multiple segments, deletions, merge, more deletions, merge again
     * WITHOUT PQ (minPqThresh high enough to disable PQ)
     * 
     * This tests the incremental graph building path with cleanup.
     * Expected behavior:
     * - First merge: Expand leading segment's graph with vectors from other segments, cleanup deleted nodes
     * - Second merge: Again expand the merged segment's graph with new vectors, cleanup deleted nodes
     */
    @Test
    public void testMultiRoundMergeWithDeletes_NoPQ() throws IOException {
        final int segmentSize = 20;
        final int numInitialSegments = 3;
        final int deletionsRound1 = 5;
        final int deletionsRound2 = 3;
        final int k = 10;
        final int minPqThresh = 1000; // High threshold to disable PQ
        final Path indexPath = createTempDir();
        final IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setUseCompoundFile(false);
        iwc.setCodec(getCodec(minPqThresh));
        iwc.setMergePolicy(new ForceMergesOnlyMergePolicy(false));

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, iwc)) {

            // Phase 1: Create initial segments
            log.info("=== Phase 1: Creating {} initial segments ===", numInitialSegments);
            for (int seg = 0; seg < numInitialSegments; seg++) {
                int docsInSegment = seg == 0 ? segmentSize : segmentSize / 2; // First segment is largest (leading)
                for (int i = 0; i < docsInSegment; i++) {
                    Document doc = new Document();
                    float[] vector = { (float) (seg * 100 + i), (float) (seg * 100 + i) * 2 };
                    doc.add(new StringField(DOC_ID_FIELD, "phase1_seg" + seg + "_doc" + i, Field.Store.YES));
                    doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.flush();
                log.info("Created segment {} with {} documents", seg, docsInSegment);
            }
            writer.commit();

            // Phase 2: Delete documents from multiple segments (including leading segment)
            log.info("=== Phase 2: Deleting {} documents from each segment ===", deletionsRound1);
            for (int seg = 0; seg < numInitialSegments; seg++) {
                int docsInSegment = seg == 0 ? segmentSize : segmentSize / 2;
                int deletionsInSegment = Math.min(deletionsRound1, docsInSegment);
                for (int i = 0; i < deletionsInSegment; i++) {
                    writer.deleteDocuments(new Term(DOC_ID_FIELD, "phase1_seg" + seg + "_doc" + i));
                }
                log.info("Deleted {} documents from segment {}", deletionsInSegment, seg);
            }
            writer.commit();

            int expectedLiveDocsAfterRound1 = segmentSize + (numInitialSegments - 1) * (segmentSize / 2) 
                - deletionsRound1 - (numInitialSegments - 1) * deletionsRound1;

            // Verify deletions before first merge
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Live docs after round 1 deletions", expectedLiveDocsAfterRound1, reader.numDocs());
                log.info("Verified {} live documents before first merge", expectedLiveDocsAfterRound1);
            }

            // Phase 3: First merge
            log.info("=== Phase 3: Performing first merge (with deletions, no PQ) ===");
            writer.forceMerge(1);
            writer.commit();

            // Verify first merge
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after first merge", 1, reader.getContext().leaves().size());
                Assert.assertEquals("Live docs after first merge", expectedLiveDocsAfterRound1, reader.numDocs());
                log.info("First merge successful: 1 segment with {} live documents", expectedLiveDocsAfterRound1);

                // Verify search works and deleted docs don't appear
                IndexSearcher searcher = newSearcher(reader);
                float[] queryVector = { 50.0f, 100.0f };
                JVectorKnnFloatVectorQuery knnQuery = createKnnQuery(TEST_FIELD, queryVector, k, new MatchAllDocsQuery());
                TopDocs topDocs = searcher.search(knnQuery, k);
                Assert.assertEquals("Should return k results", k, topDocs.totalHits.value());

                // Verify deleted documents are not in results
                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String docId = doc.get(DOC_ID_FIELD);
                    for (int seg = 0; seg < numInitialSegments; seg++) {
                        for (int j = 0; j < deletionsRound1; j++) {
                            Assert.assertNotEquals("Deleted doc should not appear", "phase1_seg" + seg + "_doc" + j, docId);
                        }
                    }
                }
                log.info("Search verification passed after first merge");
            }

            // Phase 4: Add new segments
            log.info("=== Phase 4: Adding new segments ===");
            for (int seg = 0; seg < 2; seg++) {
                for (int i = 0; i < segmentSize / 2; i++) {
                    Document doc = new Document();
                    float[] vector = { (float) (500 + seg * 100 + i), (float) (500 + seg * 100 + i) * 2 };
                    doc.add(new StringField(DOC_ID_FIELD, "phase4_seg" + seg + "_doc" + i, Field.Store.YES));
                    doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.flush();
                log.info("Added new segment {} with {} documents", seg, segmentSize / 2);
            }
            writer.commit();

            // Phase 5: Delete documents from the merged segment and new segments
            log.info("=== Phase 5: Deleting {} documents from merged and new segments ===", deletionsRound2);
            // Delete from merged segment (using phase1 doc IDs that weren't deleted in round 1)
            for (int i = deletionsRound1; i < deletionsRound1 + deletionsRound2; i++) {
                writer.deleteDocuments(new Term(DOC_ID_FIELD, "phase1_seg0_doc" + i));
            }
            // Delete from new segments
            for (int seg = 0; seg < 2; seg++) {
                for (int i = 0; i < deletionsRound2; i++) {
                    writer.deleteDocuments(new Term(DOC_ID_FIELD, "phase4_seg" + seg + "_doc" + i));
                }
            }
            writer.commit();

            int expectedLiveDocsAfterRound2 = expectedLiveDocsAfterRound1 + 2 * (segmentSize / 2) 
                - deletionsRound2 - 2 * deletionsRound2;

            // Verify deletions before second merge
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Live docs after round 2 deletions", expectedLiveDocsAfterRound2, reader.numDocs());
                log.info("Verified {} live documents before second merge", expectedLiveDocsAfterRound2);
            }

            // Phase 6: Second merge
            log.info("=== Phase 6: Performing second merge (re-merging with more deletions, no PQ) ===");
            writer.forceMerge(1);
            writer.commit();

            // Final verification
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after second merge", 1, reader.getContext().leaves().size());
                Assert.assertEquals("Final live docs count", expectedLiveDocsAfterRound2, reader.numDocs());
                log.info("Second merge successful: 1 segment with {} live documents", expectedLiveDocsAfterRound2);

                // Verify search still works correctly
                IndexSearcher searcher = newSearcher(reader);
                float[] queryVector = { 100.0f, 200.0f };
                JVectorKnnFloatVectorQuery knnQuery = createKnnQuery(TEST_FIELD, queryVector, k, new MatchAllDocsQuery());
                TopDocs topDocs = searcher.search(knnQuery, k);
                Assert.assertEquals("Should return k results after second merge", k, topDocs.totalHits.value());

                // Verify all deleted documents are excluded
                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String docId = doc.get(DOC_ID_FIELD);
                    
                    // Check round 1 deletions
                    for (int seg = 0; seg < numInitialSegments; seg++) {
                        for (int j = 0; j < deletionsRound1; j++) {
                            Assert.assertNotEquals("Round 1 deleted doc should not appear", "phase1_seg" + seg + "_doc" + j, docId);
                        }
                    }
                    
                    // Check round 2 deletions from merged segment
                    for (int j = deletionsRound1; j < deletionsRound1 + deletionsRound2; j++) {
                        Assert.assertNotEquals("Round 2 deleted doc should not appear", "phase1_seg0_doc" + j, docId);
                    }
                    
                    // Check round 2 deletions from new segments
                    for (int seg = 0; seg < 2; seg++) {
                        for (int j = 0; j < deletionsRound2; j++) {
                            Assert.assertNotEquals("Round 2 new segment deleted doc should not appear", 
                                "phase4_seg" + seg + "_doc" + j, docId);
                        }
                    }
                }
                log.info("Final search verification passed after second merge");
            }

            log.info("=== Test completed successfully: Multi-round merge with deletes (no PQ) ===");
        }
    }

    /**
     * Test Scenario 2: Multiple segments, deletions, merge, more deletions, merge again
     * WITH PQ enabled (minPqThresh low enough to trigger PQ on first merge)
     * 
     * This tests the PQ path where new codebooks are created on first merge,
     * then refined on second merge.
     * Expected behavior:
     * - First merge: Create new PQ codebooks, build graph from scratch with PQ vectors
     * - Second merge: Refine existing PQ codebooks, build graph from scratch with refined PQ vectors
     */
    @Test
    public void testMultiRoundMergeWithDeletes_WithPQ() throws IOException {
        final int segmentSize = 30;
        final int numInitialSegments = 3;
        final int deletionsRound1 = 4;
        final int deletionsRound2 = 3;
        final int k = 10;
        final int minPqThresh = 40; // Low threshold to enable PQ
        final Path indexPath = createTempDir();
        final IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setUseCompoundFile(false);
        iwc.setCodec(getCodec(minPqThresh));
        iwc.setMergePolicy(new ForceMergesOnlyMergePolicy(false));

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, iwc)) {

            // Phase 1: Create initial segments (enough vectors to trigger PQ)
            log.info("=== Phase 1: Creating {} initial segments (PQ will be triggered) ===", numInitialSegments);
            for (int seg = 0; seg < numInitialSegments; seg++) {
                int docsInSegment = seg == 0 ? segmentSize : segmentSize / 2;
                for (int i = 0; i < docsInSegment; i++) {
                    Document doc = new Document();
                    float[] vector = { (float) (seg * 100 + i), (float) (seg * 100 + i) * 2 };
                    doc.add(new StringField(DOC_ID_FIELD, "pq_phase1_seg" + seg + "_doc" + i, Field.Store.YES));
                    doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.flush();
                log.info("Created segment {} with {} documents", seg, docsInSegment);
            }
            writer.commit();

            int totalVectorsPhase1 = segmentSize + (numInitialSegments - 1) * (segmentSize / 2);
            log.info("Total vectors in phase 1: {} (threshold: {})", totalVectorsPhase1, minPqThresh);
            Assert.assertTrue("Should have enough vectors to trigger PQ", totalVectorsPhase1 >= minPqThresh);

            // Phase 2: Delete documents from multiple segments
            log.info("=== Phase 2: Deleting {} documents from each segment ===", deletionsRound1);
            for (int seg = 0; seg < numInitialSegments; seg++) {
                int docsInSegment = seg == 0 ? segmentSize : segmentSize / 2;
                int deletionsInSegment = Math.min(deletionsRound1, docsInSegment);
                for (int i = 0; i < deletionsInSegment; i++) {
                    writer.deleteDocuments(new Term(DOC_ID_FIELD, "pq_phase1_seg" + seg + "_doc" + i));
                }
                log.info("Deleted {} documents from segment {}", deletionsInSegment, seg);
            }
            writer.commit();

            int expectedLiveDocsAfterRound1 = totalVectorsPhase1 - deletionsRound1 - (numInitialSegments - 1) * deletionsRound1;

            // Phase 3: First merge (will create new PQ codebooks)
            log.info("=== Phase 3: Performing first merge (with PQ - new codebooks) ===");
            writer.forceMerge(1);
            writer.commit();

            // Verify first merge
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after first merge", 1, reader.getContext().leaves().size());
                Assert.assertEquals("Live docs after first merge", expectedLiveDocsAfterRound1, reader.numDocs());
                log.info("First merge successful with PQ: 1 segment with {} live documents", expectedLiveDocsAfterRound1);

                // Verify search works with PQ
                IndexSearcher searcher = newSearcher(reader);
                float[] queryVector = { 50.0f, 100.0f };
                JVectorKnnFloatVectorQuery knnQuery = createKnnQuery(TEST_FIELD, queryVector, k, new MatchAllDocsQuery());
                TopDocs topDocs = searcher.search(knnQuery, k);
                Assert.assertEquals("Should return k results with PQ", k, topDocs.totalHits.value());
                log.info("Search with PQ successful after first merge");
            }

            // Phase 4: Add new segments
            log.info("=== Phase 4: Adding new segments ===");
            for (int seg = 0; seg < 2; seg++) {
                for (int i = 0; i < segmentSize / 2; i++) {
                    Document doc = new Document();
                    float[] vector = { (float) (500 + seg * 100 + i), (float) (500 + seg * 100 + i) * 2 };
                    doc.add(new StringField(DOC_ID_FIELD, "pq_phase4_seg" + seg + "_doc" + i, Field.Store.YES));
                    doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.flush();
                log.info("Added new segment {} with {} documents", seg, segmentSize / 2);
            }
            writer.commit();

            // Phase 5: Delete documents from merged segment and new segments
            log.info("=== Phase 5: Deleting {} documents from merged and new segments ===", deletionsRound2);
            for (int i = deletionsRound1; i < deletionsRound1 + deletionsRound2; i++) {
                writer.deleteDocuments(new Term(DOC_ID_FIELD, "pq_phase1_seg0_doc" + i));
            }
            for (int seg = 0; seg < 2; seg++) {
                for (int i = 0; i < deletionsRound2; i++) {
                    writer.deleteDocuments(new Term(DOC_ID_FIELD, "pq_phase4_seg" + seg + "_doc" + i));
                }
            }
            writer.commit();

            int expectedLiveDocsAfterRound2 = expectedLiveDocsAfterRound1 + 2 * (segmentSize / 2) 
                - deletionsRound2 - 2 * deletionsRound2;

            // Phase 6: Second merge (will refine existing PQ codebooks)
            log.info("=== Phase 6: Performing second merge (with PQ - refined codebooks) ===");
            writer.forceMerge(1);
            writer.commit();

            // Final verification
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after second merge", 1, reader.getContext().leaves().size());
                Assert.assertEquals("Final live docs count", expectedLiveDocsAfterRound2, reader.numDocs());
                log.info("Second merge successful with refined PQ: 1 segment with {} live documents", expectedLiveDocsAfterRound2);

                // Verify search still works with refined PQ
                IndexSearcher searcher = newSearcher(reader);
                float[] queryVector = { 100.0f, 200.0f };
                JVectorKnnFloatVectorQuery knnQuery = createKnnQuery(TEST_FIELD, queryVector, k, new MatchAllDocsQuery());
                TopDocs topDocs = searcher.search(knnQuery, k);
                Assert.assertEquals("Should return k results with refined PQ", k, topDocs.totalHits.value());
                log.info("Search with refined PQ successful after second merge");
            }

            log.info("=== Test completed successfully: Multi-round merge with deletes (with PQ) ===");
        }
    }

    /**
     * Test Scenario 3: Transition from no PQ to PQ
     * First merge without PQ (below threshold), second merge with PQ (above threshold)
     * 
     * This tests the transition where the first merge uses incremental graph building,
     * but the second merge has enough vectors to trigger PQ.
     */
    @Test
    public void testMergeTransition_NoPQToPQ() throws IOException {
        final int smallSegmentSize = 15;
        final int largeSegmentSize = 30;
        final int deletions = 2;
        final int k = 10;
        final int minPqThresh = 40;
        final Path indexPath = createTempDir();
        final IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setUseCompoundFile(false);
        iwc.setCodec(getCodec(minPqThresh));
        iwc.setMergePolicy(new ForceMergesOnlyMergePolicy(false));

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, iwc)) {

            // Phase 1: Create small segments (below PQ threshold)
            log.info("=== Phase 1: Creating small segments (below PQ threshold) ===");
            for (int seg = 0; seg < 2; seg++) {
                for (int i = 0; i < smallSegmentSize; i++) {
                    Document doc = new Document();
                    float[] vector = { (float) (seg * 100 + i), (float) (seg * 100 + i) * 2 };
                    doc.add(new StringField(DOC_ID_FIELD, "trans_phase1_seg" + seg + "_doc" + i, Field.Store.YES));
                    doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.flush();
            }
            writer.commit();

            int totalVectorsPhase1 = 2 * smallSegmentSize;
            log.info("Total vectors in phase 1: {} (threshold: {})", totalVectorsPhase1, minPqThresh);
            Assert.assertTrue("Should be below PQ threshold", totalVectorsPhase1 < minPqThresh);

            // Delete some documents
            for (int i = 0; i < deletions; i++) {
                writer.deleteDocuments(new Term(DOC_ID_FIELD, "trans_phase1_seg0_doc" + i));
            }
            writer.commit();

            // Phase 2: First merge (no PQ - below threshold)
            log.info("=== Phase 2: First merge (no PQ - below threshold) ===");
            writer.forceMerge(1);
            writer.commit();

            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment", 1, reader.getContext().leaves().size());
                log.info("First merge completed without PQ");
            }

            // Phase 3: Add large segments (will push total above PQ threshold)
            log.info("=== Phase 3: Adding large segments (will trigger PQ on next merge) ===");
            for (int seg = 0; seg < 2; seg++) {
                for (int i = 0; i < largeSegmentSize; i++) {
                    Document doc = new Document();
                    float[] vector = { (float) (500 + seg * 100 + i), (float) (500 + seg * 100 + i) * 2 };
                    doc.add(new StringField(DOC_ID_FIELD, "trans_phase3_seg" + seg + "_doc" + i, Field.Store.YES));
                    doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.flush();
            }
            writer.commit();

            int totalVectorsPhase3 = totalVectorsPhase1 - deletions + 2 * largeSegmentSize;
            log.info("Total vectors before second merge: {} (threshold: {})", totalVectorsPhase3, minPqThresh);
            Assert.assertTrue("Should be above PQ threshold", totalVectorsPhase3 >= minPqThresh);

            // Delete some more documents
            for (int i = 0; i < deletions; i++) {
                writer.deleteDocuments(new Term(DOC_ID_FIELD, "trans_phase3_seg0_doc" + i));
            }
            writer.commit();

            // Phase 4: Second merge (with PQ - above threshold)
            log.info("=== Phase 4: Second merge (with PQ - above threshold) ===");
            writer.forceMerge(1);
            writer.commit();

            // Final verification
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment", 1, reader.getContext().leaves().size());
                int expectedFinalDocs = totalVectorsPhase3 - deletions;
                Assert.assertEquals("Final live docs count", expectedFinalDocs, reader.numDocs());
                log.info("Second merge completed with PQ: {} live documents", expectedFinalDocs);

                // Verify search works
                IndexSearcher searcher = newSearcher(reader);
                float[] queryVector = { 100.0f, 200.0f };
                JVectorKnnFloatVectorQuery knnQuery = createKnnQuery(TEST_FIELD, queryVector, k, new MatchAllDocsQuery());
                TopDocs topDocs = searcher.search(knnQuery, k);
                Assert.assertEquals("Should return k results", k, topDocs.totalHits.value());
                log.info("Search successful after transition to PQ");
            }

            log.info("=== Test completed successfully: Transition from no PQ to PQ ===");
        }
    }

    /**
     * Test Scenario 4: All documents deleted in leading segment
     * Tests edge case where the leading segment has all documents deleted before merge.
     */
    @Test
    public void testMergeWithAllDeletesInLeadingSegment_MultipleMerges() throws IOException {
        final int segmentSize = 20;
        final int k = 5;
        final int minPqThresh = 100; // Disable PQ
        final Path indexPath = createTempDir();
        final IndexWriterConfig iwc = newIndexWriterConfig();
        iwc.setUseCompoundFile(false);
        iwc.setCodec(getCodec(minPqThresh));
        iwc.setMergePolicy(new ForceMergesOnlyMergePolicy(false));

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, iwc)) {

            // Phase 1: Create segments
            log.info("=== Phase 1: Creating segments ===");
            for (int seg = 0; seg < 3; seg++) {
                int docsInSegment = seg == 0 ? segmentSize : segmentSize / 2;
                for (int i = 0; i < docsInSegment; i++) {
                    Document doc = new Document();
                    float[] vector = { (float) (seg * 100 + i), (float) (seg * 100 + i) * 2 };
                    doc.add(new StringField(DOC_ID_FIELD, "alldel_seg" + seg + "_doc" + i, Field.Store.YES));
                    doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.flush();
            }
            writer.commit();

            // Phase 2: Delete ALL documents from leading segment
            log.info("=== Phase 2: Deleting ALL documents from leading segment ===");
            for (int i = 0; i < segmentSize; i++) {
                writer.deleteDocuments(new Term(DOC_ID_FIELD, "alldel_seg0_doc" + i));
            }
            writer.commit();

            int expectedLiveDocs = 2 * (segmentSize / 2);

            // Phase 3: First merge
            log.info("=== Phase 3: First merge with all leading segment docs deleted ===");
            writer.forceMerge(1);
            writer.commit();

            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment", 1, reader.getContext().leaves().size());
                Assert.assertEquals("Only non-leading segment docs should remain", expectedLiveDocs, reader.numDocs());
                log.info("First merge successful with {} live documents", expectedLiveDocs);
            }

            // Phase 4: Add more segments
            log.info("=== Phase 4: Adding new segments ===");
            for (int i = 0; i < segmentSize / 2; i++) {
                Document doc = new Document();
                float[] vector = { (float) (500 + i), (float) (500 + i) * 2 };
                doc.add(new StringField(DOC_ID_FIELD, "alldel_new_doc" + i, Field.Store.YES));
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc);
            }
            writer.flush();
            writer.commit();

            // Phase 5: Delete some from merged segment
            log.info("=== Phase 5: Deleting some documents from merged segment ===");
            for (int i = 0; i < 3; i++) {
                writer.deleteDocuments(new Term(DOC_ID_FIELD, "alldel_seg1_doc" + i));
            }
            writer.commit();

            // Phase 6: Second merge
            log.info("=== Phase 6: Second merge ===");
            writer.forceMerge(1);
            writer.commit();

            int expectedFinalDocs = expectedLiveDocs + segmentSize / 2 - 3;

            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment", 1, reader.getContext().leaves().size());
                Assert.assertEquals("Final live docs count", expectedFinalDocs, reader.numDocs());
                log.info("Second merge successful with {} live documents", expectedFinalDocs);

                // Verify search works
                IndexSearcher searcher = newSearcher(reader);
                float[] queryVector = { 200.0f, 400.0f };
                JVectorKnnFloatVectorQuery knnQuery = createKnnQuery(TEST_FIELD, queryVector, k, new MatchAllDocsQuery());
                TopDocs topDocs = searcher.search(knnQuery, k);
                Assert.assertEquals("Should return k results", k, topDocs.totalHits.value());
            }

            log.info("=== Test completed successfully: All deletes in leading segment with multiple merges ===");
        }
    }

    /**
     * Helper method to create a KNN query.
     */
    private JVectorKnnFloatVectorQuery createKnnQuery(String fieldName, float[] target, int k, Query filterQuery) {
        return new JVectorKnnFloatVectorQuery(
            fieldName,
            target,
            k,
            filterQuery,
            KNNConstants.DEFAULT_OVER_QUERY_FACTOR,
            KNNConstants.DEFAULT_QUERY_SIMILARITY_THRESHOLD.floatValue(),
            KNNConstants.DEFAULT_QUERY_RERANK_FLOOR.floatValue(),
            KNNConstants.DEFAULT_QUERY_USE_PRUNING
        );
    }
}

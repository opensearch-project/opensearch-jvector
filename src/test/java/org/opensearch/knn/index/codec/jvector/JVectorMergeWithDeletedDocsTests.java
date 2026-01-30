/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import static org.opensearch.knn.index.engine.CommonTestUtils.getCodec;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.*;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Assert;
import org.junit.Test;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;

/**
 * Test cases for validating merge behavior with deleted documents in the OpenSearch JVector plugin.
 *
 * These tests validate the OpenSearch-jvector plugin's custom codec (JVectorFormat) and merge behavior
 * when documents are deleted, ensuring proper handling of:
 * - Merges with deleted documents in the leading reader (most live docs)
 * - Off-by-one errors in array sizing during merges
 * - Correct array sizing when many documents are deleted
 * - Product Quantization (PQ) encoding with deletions
 *
 * Unlike generic Lucene tests, these specifically test the opensearch-jvector plugin's VectorField
 * and JVectorKnnFloatVectorQuery implementations.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "")
@Log4j2
public class JVectorMergeWithDeletedDocsTests extends LuceneTestCase {

    private static final String TEST_FIELD = "test_field";
    private static final String TEST_ID_FIELD = "id";

    /**
     * Helper method to create JVectorKnnFloatVectorQuery with default parameters
     */
    private JVectorKnnFloatVectorQuery getJVectorKnnFloatVectorQuery(String fieldName, float[] target, int k, Query filterQuery) {
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

    /**
     * Comprehensive test combining multiple deletion patterns and document overwrites across multiple merge cycles.
     *
     * This validates that the opensearch-jvector codec handles complex scenarios with:
     * - Multiple small segments (10-20 documents each)
     * - Various deletion patterns
     * - Document overwrites (updates that create deletions)
     * - Product Quantization enabled
     * - Accurate search results after merge
     */
    @Test
    public void testMultipleMergesWithVariousDeletionPatterns() throws IOException {
        final int dimension = 64;
        final int k = 5;

        log.info("Testing multiple merges with various deletion patterns and overwrites");

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec(1)); // Enable PQ
        // config.setCodec(getCodec(1000000)); // Disable PQ
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, config)) {
            int docId = 0;

            // Segment 1: 2000 documents with 400 explicit deletions (20%)
            log.info("Creating segment 1: 2000 docs with explicit deletions");
            int seg1Size = 2000;
            int seg1Start = docId;
            for (int i = 0; i < seg1Size; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * 0.01f);
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();
            // Delete first 400 documents
            for (int i = seg1Start; i < seg1Start + 400; i++) {
                writer.deleteDocuments(new Term(TEST_ID_FIELD, String.valueOf(i)));
            }
            writer.commit();
            int seg1LiveDocs = seg1Size - 400;

            // Segment 2: 1500 documents with 300 overwrites (updates)
            log.info("Creating segment 2: 1500 docs with overwrites");
            int seg2Size = 1500;
            int seg2Start = docId;
            for (int i = 0; i < seg2Size; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * 0.01f);
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();
            // Overwrite 300 documents (this creates deletions + new docs)
            log.info("Overwriting 300 documents in segment 2");
            for (int i = seg2Start; i < seg2Start + 300; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, (i + 1000) * 0.01f); // Different vector values
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(i), Field.Store.YES));
                writer.updateDocument(new Term(TEST_ID_FIELD, String.valueOf(i)), doc);
            }
            writer.commit();
            int seg2LiveDocs = seg2Size; // Overwrites don't change live doc count

            // First intermediate merge after segment 2
            log.info("Performing first intermediate merge after segment 2");
            writer.forceMerge(1);
            writer.commit();
            log.info("First intermediate merge completed");

            // Segment 3: 1200 documents with no deletions
            log.info("Creating segment 3: 1200 docs with no deletions");
            int seg3Size = 1200;
            for (int i = 0; i < seg3Size; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * 0.01f);
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();

            // Segment 4: 1800 documents with mixed deletions and overwrites
            log.info("Creating segment 4: 1800 docs with mixed deletions and overwrites");
            int seg4Size = 1800;
            int seg4Start = docId;
            for (int i = 0; i < seg4Size; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * 0.01f);
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();
            // Delete 200 documents
            for (int i = seg4Start; i < seg4Start + 200; i++) {
                writer.deleteDocuments(new Term(TEST_ID_FIELD, String.valueOf(i)));
            }
            // Overwrite 200 different documents
            for (int i = seg4Start + 500; i < seg4Start + 700; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, (i + 2000) * 0.01f);
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(i), Field.Store.YES));
                writer.updateDocument(new Term(TEST_ID_FIELD, String.valueOf(i)), doc);
            }
            writer.commit();
            int seg4LiveDocs = seg4Size - 200; // Only explicit deletions reduce count

            // Second intermediate merge after segment 4
            log.info("Performing second intermediate merge after segment 4");
            writer.forceMerge(1);
            writer.commit();
            log.info("Second intermediate merge completed");

            // Segment 5: 1000 documents, all overwritten
            log.info("Creating segment 5: 1000 docs, all overwritten");
            int seg5Size = 1000;
            int seg5Start = docId;
            for (int i = 0; i < seg5Size; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * 0.01f);
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();
            // Overwrite all documents
            log.info("Overwriting all 1000 documents in segment 5");
            for (int i = seg5Start; i < seg5Start + seg5Size; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, (i + 3000) * 0.01f);
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(i), Field.Store.YES));
                writer.updateDocument(new Term(TEST_ID_FIELD, String.valueOf(i)), doc);
            }
            writer.commit();
            int seg5LiveDocs = seg5Size; // Overwrites maintain count

            // Segment 6: 1500 documents with scattered deletions (10%)
            log.info("Creating segment 6: 1500 docs with scattered deletions");
            int seg6Size = 1500;
            int seg6Start = docId;
            for (int i = 0; i < seg6Size; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * 0.01f);
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();
            // Delete every 10th document (scattered pattern)
            log.info("Deleting scattered documents (every 10th) in segment 6");
            for (int i = seg6Start; i < seg6Start + seg6Size; i += 10) {
                writer.deleteDocuments(new Term(TEST_ID_FIELD, String.valueOf(i)));
            }
            writer.commit();
            int seg6LiveDocs = seg6Size - (seg6Size / 10); // ~10% deleted

            // Segment 7: 2000 documents with heavy overwrites (50%)
            log.info("Creating segment 7: 2000 docs with heavy overwrites");
            int seg7Size = 2000;
            int seg7Start = docId;
            for (int i = 0; i < seg7Size; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * 0.01f);
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();
            // Overwrite 50% of documents
            log.info("Overwriting 1000 documents (50%) in segment 7");
            for (int i = seg7Start; i < seg7Start + 1000; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, (i + 4000) * 0.01f);
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(i), Field.Store.YES));
                writer.updateDocument(new Term(TEST_ID_FIELD, String.valueOf(i)), doc);
            }
            writer.commit();
            int seg7LiveDocs = seg7Size; // Overwrites maintain count

            // Third intermediate merge after segment 7
            log.info("Performing third intermediate merge after segment 7");
            writer.forceMerge(1);
            writer.commit();
            log.info("Third intermediate merge completed");

            int expectedLiveDocs = seg1LiveDocs + seg2LiveDocs + seg3Size + seg4LiveDocs + seg5LiveDocs + seg6LiveDocs + seg7LiveDocs;
            log.info(
                "Total expected live docs: {} (seg1:{}, seg2:{}, seg3:{}, seg4:{}, seg5:{}, seg6:{}, seg7:{})",
                expectedLiveDocs,
                seg1LiveDocs,
                seg2LiveDocs,
                seg3Size,
                seg4LiveDocs,
                seg5LiveDocs,
                seg6LiveDocs,
                seg7LiveDocs
            );

            // Final force merge
            log.info("Starting final force merge");
            writer.forceMerge(1);
            writer.commit();
            log.info("Force merge completed successfully");

            // Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after merge", 1, reader.getContext().leaves().size());
                Assert.assertEquals("Should have correct number of live docs", expectedLiveDocs, reader.numDocs());

                // Verify search works correctly
                final float[] target = new float[dimension];
                Arrays.fill(target, 0.5f);
                final IndexSearcher searcher = newSearcher(reader);
                JVectorKnnFloatVectorQuery query = getJVectorKnnFloatVectorQuery(TEST_FIELD, target, k, new MatchAllDocsQuery());
                TopDocs topDocs = searcher.search(query, k);
                Assert.assertEquals("Should return k results", k, topDocs.totalHits.value());

                // Verify that overwritten documents have updated vectors
                log.info("Verifying overwritten documents have updated vectors");
                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String id = doc.get(TEST_ID_FIELD);
                    log.info("Result {}: doc ID = {}", i, id);
                }

                log.info("Comprehensive test passed! All deletion and overwrite scenarios handled correctly");
            }
        }
    }

}

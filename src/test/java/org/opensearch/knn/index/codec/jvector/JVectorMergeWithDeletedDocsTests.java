/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import static org.opensearch.knn.index.engine.CommonTestUtils.getCodec;
import static org.hamcrest.Matchers.equalTo;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.*;
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
     * Comprehensive test combining merges with one document that
     * have no vector fields populated, multiple segments.
     */
    @Test
    public void testMergesWithOneDeleteAndOneNullVector() throws IOException {
        final Map<String, float[]> docs = new HashMap<>();
        final int dimension = 64;
        final int k = 4;

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec(1));
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, config)) {
            int docId = 0;

            // Segment 1: Updates document with no vector field populated
            for (int i = 0; i < 1; i++) {
                Document doc = new Document();
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                writer.addDocument(doc);
                docId++;
            }

            // Segment 1: add 3 documents
            log.info("Creating segment 1: 3 docs");
            for (int i = 0; i < 3; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * random().nextFloat(1.0f));
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                docs.put(Integer.toString(docId), vector);
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();

            // Segment 2: Add one more document
            for (int i = 0; i < 1; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * random().nextFloat(1.0f));
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                docs.put(Integer.toString(docId), vector);
                writer.addDocument(doc);
                docId++;

            }
            // Delete one document
            writer.deleteDocuments(new Term(TEST_ID_FIELD, String.valueOf(2)));
            writer.commit();

            log.info("Performing intermediate merge after segment 2");
            writer.forceMerge(1);

            // Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after merge", 1, reader.getContext().leaves().size());
                Assert.assertEquals("Should have correct number of live docs", 4, reader.numDocs());

                // Verify search works correctly
                final IndexSearcher searcher = newSearcher(reader);
                TopDocs topDocs = searcher.search(new MatchAllDocsQuery(), k);
                Assert.assertEquals("Should return k results", k, topDocs.totalHits.value());

                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String id = doc.get(TEST_ID_FIELD);
                    assertThat(getVector(reader, TEST_FIELD, topDocs.scoreDocs[i].doc), equalTo(docs.get(id)));
                    log.info("Result {}: doc ID = {}", i, id);
                }
            }
        }
    }

    /**
     * Comprehensive test combining merges with one document that
     * have no vector fields populated, multiple segments.
     */
    @Test
    public void testMergesWithOneNonNullVector() throws IOException {
        final Map<String, float[]> docs = new HashMap<>();
        final int dimension = 64;
        final int k = 5;

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec(1));
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, config)) {
            int docId = 0;
            int docIdWithVector = random().nextInt(4);

            // Segment 1: Add 3 documents with no vector and one with the vector
            for (int i = 0; i < 4; i++) {
                Document doc = new Document();

                if (docId == docIdWithVector) {
                    float[] vector = new float[dimension];
                    Arrays.fill(vector, docId * random().nextFloat(1.0f));
                    doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                    docs.put(Integer.toString(docId), vector);
                }

                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();

            // Segment 2: Add one more document with no vectors
            for (int i = 0; i < 1; i++) {
                Document doc = new Document();
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                writer.addDocument(doc);
                docId++;

            }
            writer.commit();

            log.info("Performing intermediate merge after segment 2");
            writer.forceMerge(1);

            // Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after merge", 1, reader.getContext().leaves().size());
                Assert.assertEquals("Should have correct number of live docs", 5, reader.numDocs());

                // Verify search works correctly
                final IndexSearcher searcher = newSearcher(reader);
                TopDocs topDocs = searcher.search(new MatchAllDocsQuery(), k);
                Assert.assertEquals("Should return k results", k, topDocs.totalHits.value());

                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String id = doc.get(TEST_ID_FIELD);
                    assertThat(getVector(reader, TEST_FIELD, topDocs.scoreDocs[i].doc), equalTo(docs.get(id)));
                    log.info("Result {}: doc ID = {}", i, id);
                }
            }
        }
    }

    /**
     * Comprehensive test combining merges with one document that
     * have no vector fields populated, multiple segments.
     */
    @Test
    public void testMergesWithOneNullVector() throws IOException {
        final Map<String, float[]> docs = new HashMap<>();
        final int dimension = 64;
        final int k = 5;

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec(1));
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, config)) {
            int docId = 0;

            // Segment 1: Updates document with no vector field populated
            for (int i = 0; i < 1; i++) {
                Document doc = new Document();
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                writer.addDocument(doc);
                docId++;
            }

            // Segment 1: add 3 documents
            log.info("Creating segment 1: 3 docs");
            for (int i = 0; i < 3; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * random().nextFloat(1.0f));
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                docs.put(Integer.toString(docId), vector);
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();

            // Segment 2: Add one more document
            for (int i = 0; i < 1; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * random().nextFloat(1.0f));
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                docs.put(Integer.toString(docId), vector);
                writer.addDocument(doc);
                docId++;

            }
            writer.commit();

            log.info("Performing intermediate merge after segment 2");
            writer.forceMerge(1);

            // Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after merge", 1, reader.getContext().leaves().size());
                Assert.assertEquals("Should have correct number of live docs", 5, reader.numDocs());

                // Verify search works correctly
                final IndexSearcher searcher = newSearcher(reader);
                TopDocs topDocs = searcher.search(new MatchAllDocsQuery(), k);
                Assert.assertEquals("Should return k results", k, topDocs.totalHits.value());

                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String id = doc.get(TEST_ID_FIELD);
                    assertThat(getVector(reader, TEST_FIELD, topDocs.scoreDocs[i].doc), equalTo(docs.get(id)));
                    log.info("Result {}: doc ID = {}", i, id);
                }
            }
        }

    }

    /**
     * Comprehensive test combining merges with documents that
     * have no vector fields populated, multiple segments where leading is not the first one.
     */
    @Test
    public void testMergesWithNullVectorsAndLastLeadingSegment() throws IOException {
        final Map<String, float[]> docs = new HashMap<>();
        final int dimension = 64;
        final int k = 9;

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec(1));
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, config)) {
            int docId = 0;

            // Segment 1: 3 documents with one document having no vector field populated
            log.info("Creating segment 1: 3 docs");
            for (int i = 0; i < 2; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * random().nextFloat(1.0f));
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                docs.put(Integer.toString(docId), vector);
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();

            // Segment 2: Add document with no vector field populated
            for (int i = 2; i < 3; i++) {
                Document doc = new Document();
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();

            // Segment 3: 6 document
            log.info("Creating segment 3: 6 docs");
            for (int i = 4; i < 10; i++) {
                Document doc = new Document();
                float[] vector = new float[dimension];
                Arrays.fill(vector, docId * random().nextFloat(1.0f));
                doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                docs.put(Integer.toString(docId), vector);
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();

            log.info("Performing intermediate merge after segment 2");
            writer.forceMerge(1);

            // Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after merge", 1, reader.getContext().leaves().size());
                Assert.assertEquals("Should have correct number of live docs", 9, reader.numDocs());

                // Verify search works correctly
                final IndexSearcher searcher = newSearcher(reader);
                TopDocs topDocs = searcher.search(new MatchAllDocsQuery(), k);
                Assert.assertEquals("Should return k results", k, topDocs.totalHits.value());

                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String id = doc.get(TEST_ID_FIELD);
                    assertThat(getVector(reader, TEST_FIELD, topDocs.scoreDocs[i].doc), equalTo(docs.get(id)));
                    log.info("Result {}: doc ID = {}", i, id);
                }
            }
        }
    }

    /**
     * Comprehensive test combining merges with documents that
     * have no vector fields populated.
     */
    @Test
    public void testMergesWithNullVectors() throws IOException {
        final Map<String, float[]> docs = new HashMap<>();
        final int dimension = 64;
        final int k = 2;

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec(1));
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, config)) {
            int docId = 0;

            // Segment 1: 3 documents with one document having no vector field populated
            log.info("Creating segment 1: 3 docs");
            for (int i = 0; i < 3; i++) {
                Document doc = new Document();
                if (i != 1) {
                    float[] vector = new float[dimension];
                    Arrays.fill(vector, docId * random().nextFloat(1.0f));
                    doc.add(new KnnFloatVectorField(TEST_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                    doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                    docs.put(Integer.toString(docId), vector);
                } else {
                    doc.add(new StringField(TEST_ID_FIELD, String.valueOf(docId), Field.Store.YES));
                }
                writer.addDocument(doc);
                docId++;
            }
            writer.commit();

            // Segment 2: Delete document with no vector field populated
            writer.deleteDocuments(new Term(TEST_ID_FIELD, String.valueOf(1)));
            writer.commit();

            log.info("Performing intermediate merge after segment 2");
            writer.forceMerge(1);

            // Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after merge", 1, reader.getContext().leaves().size());
                Assert.assertEquals("Should have correct number of live docs", 2, reader.numDocs());

                // Verify search works correctly
                final float[] target = new float[dimension];
                Arrays.fill(target, 0.5f);
                final IndexSearcher searcher = newSearcher(reader);
                JVectorKnnFloatVectorQuery query = getJVectorKnnFloatVectorQuery(TEST_FIELD, target, k, new MatchAllDocsQuery());
                TopDocs topDocs = searcher.search(query, k);
                Assert.assertEquals("Should return k results", k, topDocs.totalHits.value());

                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String id = doc.get(TEST_ID_FIELD);
                    assertThat(getVector(reader, TEST_FIELD, topDocs.scoreDocs[i].doc), equalTo(docs.get(id)));
                    log.info("Result {}: doc ID = {}", i, id);
                }
            }
        }
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

    /**
     * Test merges with root-level vector field and nested child documents.
     *
     * This test validates the scenario where:
     * - Vector field ("embedding") is at the root (parent) document level
     * - Nested child documents ("authors") contain metadata without vectors
     * - Documents are merged across multiple segments with deletions
     *
     * In Lucene, nested documents are stored as:
     * - Child documents come first (authors)
     * - Parent document comes last (with vector)
     * - All are added as a single block using addDocuments()
     *
     * This mirrors the OpenSearch index structure:
     * {
     *   "embedding": [vector],
     *   "authors": [
     *     {"name": "Author 1"},
     *     {"name": "Author 2"}
     *   ]
     * }
     */
    @Test
    public void testMergesWithRootVectorAndNestedChildren() throws IOException {
        final int dimension = 128;
        final int k = 147;
        final String VECTOR_FIELD = "embedding";
        final String AUTHOR_NAME_FIELD = "authors.name";
        final String DOC_TYPE_FIELD = "type"; // To distinguish parent from child docs

        log.info("Testing merges with root-level vectors and nested child documents");

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec(1));
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, config)) {
            int parentDocId = 0;

            // Segment 1: 50 parent documents, each with 1-3 nested child documents
            log.info("Creating segment 1: 50 parent docs with nested children");
            for (int i = 0; i < 50; i++) {
                // Create nested structure: children first, then parent
                int numChildren = 1 + (i % 3); // 1-3 children per parent
                Document[] docBlock = new Document[numChildren + 1];

                // Add child documents (authors)
                for (int c = 0; c < numChildren; c++) {
                    Document childDoc = new Document();
                    childDoc.add(new StringField(DOC_TYPE_FIELD, "nested", Field.Store.YES));
                    childDoc.add(new TextField(AUTHOR_NAME_FIELD, "Author_" + parentDocId + "_" + c, Field.Store.YES));
                    childDoc.add(new StringField("parent_id", String.valueOf(parentDocId), Field.Store.YES));
                    docBlock[c] = childDoc;
                }

                // Add parent document with vector (must be last in block)
                Document parentDoc = new Document();
                float[] vector = new float[dimension];
                for (int d = 0; d < dimension; d++) {
                    vector[d] = (parentDocId * 0.01f) + (d * 0.001f);
                }
                parentDoc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, VectorSimilarityFunction.COSINE));
                parentDoc.add(new StringField(TEST_ID_FIELD, String.valueOf(parentDocId), Field.Store.YES));
                parentDoc.add(new StringField(DOC_TYPE_FIELD, "parent", Field.Store.YES));
                parentDoc.add(new NumericDocValuesField("num_children", numChildren));
                docBlock[numChildren] = parentDoc;

                // Add the entire block (children + parent) atomically
                writer.addDocuments(Arrays.asList(docBlock));
                parentDocId++;
            }
            writer.commit();

            // Delete 10 parent documents (and their children) from segment 1
            log.info("Deleting 10 parent documents from segment 1");
            for (int i = 5; i < 15; i++) {
                writer.deleteDocuments(new Term(TEST_ID_FIELD, String.valueOf(i)));
            }
            writer.commit();

            // Segment 2: 75 parent documents with nested children
            log.info("Creating segment 2: 75 parent docs with nested children");
            for (int i = 0; i < 75; i++) {
                int numChildren = 2 + (i % 3); // 2-4 children per parent
                Document[] docBlock = new Document[numChildren + 1];

                // Child documents
                for (int c = 0; c < numChildren; c++) {
                    Document childDoc = new Document();
                    childDoc.add(new StringField(DOC_TYPE_FIELD, "child", Field.Store.YES));
                    childDoc.add(new TextField(AUTHOR_NAME_FIELD, "Author_" + parentDocId + "_" + c, Field.Store.YES));
                    childDoc.add(new StringField("parent_id", String.valueOf(parentDocId), Field.Store.YES));
                    docBlock[c] = childDoc;
                }

                // Parent document with vector
                Document parentDoc = new Document();
                float[] vector = new float[dimension];
                for (int d = 0; d < dimension; d++) {
                    vector[d] = (parentDocId * 0.01f) + (d * 0.001f);
                }
                parentDoc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, VectorSimilarityFunction.COSINE));
                parentDoc.add(new StringField(TEST_ID_FIELD, String.valueOf(parentDocId), Field.Store.YES));
                parentDoc.add(new StringField(DOC_TYPE_FIELD, "parent", Field.Store.YES));
                parentDoc.add(new NumericDocValuesField("num_children", numChildren));
                docBlock[numChildren] = parentDoc;

                writer.addDocuments(Arrays.asList(docBlock));
                parentDocId++;
            }
            writer.commit();

            // Update 15 parent documents in segment 2
            log.info("Updating 15 parent documents in segment 2");
            for (int i = 50; i < 65; i++) {
                int numChildren = 2;
                Document[] docBlock = new Document[numChildren + 1];

                for (int c = 0; c < numChildren; c++) {
                    Document childDoc = new Document();
                    childDoc.add(new StringField(DOC_TYPE_FIELD, "child", Field.Store.YES));
                    childDoc.add(new TextField(AUTHOR_NAME_FIELD, "UpdatedAuthor_" + i + "_" + c, Field.Store.YES));
                    childDoc.add(new StringField("parent_id", String.valueOf(i), Field.Store.YES));
                    docBlock[c] = childDoc;
                }

                Document parentDoc = new Document();
                float[] vector = new float[dimension];
                for (int d = 0; d < dimension; d++) {
                    vector[d] = ((i + 1000) * 0.01f) + (d * 0.001f);
                }
                parentDoc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, VectorSimilarityFunction.COSINE));
                parentDoc.add(new StringField(TEST_ID_FIELD, String.valueOf(i), Field.Store.YES));
                parentDoc.add(new StringField(DOC_TYPE_FIELD, "parent", Field.Store.YES));
                parentDoc.add(new NumericDocValuesField("num_children", numChildren));
                docBlock[numChildren] = parentDoc;

                writer.updateDocuments(new Term(TEST_ID_FIELD, String.valueOf(i)), Arrays.asList(docBlock));
            }
            writer.commit();

            // First intermediate merge
            log.info("Performing first intermediate merge");
            writer.forceMerge(1);

            // Segment 3: 40 parent documents
            log.info("Creating segment 3: 40 parent docs with nested children");
            for (int i = 0; i < 40; i++) {
                int numChildren = 1 + (i % 4); // 1-4 children
                Document[] docBlock = new Document[numChildren + 1];

                for (int c = 0; c < numChildren; c++) {
                    Document childDoc = new Document();
                    childDoc.add(new StringField(DOC_TYPE_FIELD, "child", Field.Store.YES));
                    childDoc.add(new TextField(AUTHOR_NAME_FIELD, "Author_" + parentDocId + "_" + c, Field.Store.YES));
                    childDoc.add(new StringField("parent_id", String.valueOf(parentDocId), Field.Store.YES));
                    docBlock[c] = childDoc;
                }

                Document parentDoc = new Document();
                float[] vector = new float[dimension];
                for (int d = 0; d < dimension; d++) {
                    vector[d] = (parentDocId * 0.01f) + (d * 0.001f);
                }
                parentDoc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, VectorSimilarityFunction.COSINE));
                parentDoc.add(new StringField(TEST_ID_FIELD, String.valueOf(parentDocId), Field.Store.YES));
                parentDoc.add(new StringField(DOC_TYPE_FIELD, "parent", Field.Store.YES));
                parentDoc.add(new NumericDocValuesField("num_children", numChildren));
                docBlock[numChildren] = parentDoc;

                writer.addDocuments(Arrays.asList(docBlock));
                parentDocId++;
            }
            writer.commit();

            // Delete 8 parent documents from segment 3
            log.info("Deleting 8 parent documents from segment 3");
            for (int i = 125; i < 133; i++) {
                writer.deleteDocuments(new Term(TEST_ID_FIELD, String.valueOf(i)));
            }
            writer.commit();

            // Calculate expected parent documents
            // Segment 1: 50 - 10 deleted = 40 parents
            // Segment 2: 75 parents (updates don't change count)
            // Segment 3: 40 - 8 deleted = 32 parents
            int expectedParents = 40 + 75 + 32;
            log.info("Expected parent documents: {}", expectedParents);

            // Final force merge
            log.info("Starting final force merge");
            writer.forceMerge(1);

            // Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after final merge", 1, reader.getContext().leaves().size());

                // Count parent documents
                IndexSearcher searcher = newSearcher(reader);
                TopDocs parentDocs = searcher.search(new TermQuery(new Term(DOC_TYPE_FIELD, "parent")), Integer.MAX_VALUE);
                Assert.assertEquals("Should have correct number of parent docs", expectedParents, parentDocs.totalHits.value());

                // Verify vector search works on parent documents
                log.info("Verifying vector search on merged index with nested structure");
                final float[] target = new float[dimension];
                Arrays.fill(target, 0.5f);
                JVectorKnnFloatVectorQuery query = getJVectorKnnFloatVectorQuery(
                    VECTOR_FIELD,
                    target,
                    k,
                    new TermQuery(new Term(DOC_TYPE_FIELD, "parent")) // Only search parent docs
                );
                TopDocs topDocs = searcher.search(query, k);
                Assert.assertEquals("Should return k results", k, topDocs.totalHits.value());

                // Verify results are parent documents with vectors
                log.info("Verifying search results are parent documents");
                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String docType = doc.get(DOC_TYPE_FIELD);
                    String id = doc.get(TEST_ID_FIELD);

                    Assert.assertEquals("Result should be a parent document", "parent", docType);
                    assertNotNull("Parent document should have ID", id);

                    log.info("Result {}: parent doc ID = {}, score = {}", i, id, topDocs.scoreDocs[i].score);
                }

                // Verify deleted parent documents are not in results
                log.info("Verifying deleted parent documents are not in index");
                for (int deletedId = 5; deletedId < 15; deletedId++) {
                    TopDocs deletedDocs = searcher.search(
                        new BooleanQuery.Builder().add(
                            new TermQuery(new Term(TEST_ID_FIELD, String.valueOf(deletedId))),
                            BooleanClause.Occur.MUST
                        ).add(new TermQuery(new Term(DOC_TYPE_FIELD, "parent")), BooleanClause.Occur.MUST).build(),
                        1
                    );
                    Assert.assertEquals("Deleted parent document " + deletedId + " should not be found", 0, deletedDocs.totalHits.value());
                }

                log.info("Test passed! Root vectors with nested children handled correctly during merge");
            }
        }
    }

    /**
     * Test merges with root-level vector field and nested child documents,
     * one of the documents has root-level vector field set to null.
     *
     * This test validates the scenario where:
     * - Vector field ("embedding") is at the root (parent) document level
     * - Nested child documents ("authors") contain metadata without vectors
     * - Documents are merged across multiple segments with deletions
     *
     * In Lucene, nested documents are stored as:
     * - Child documents come first (authors)
     * - Parent document comes last (with vector)
     * - All are added as a single block using addDocuments()
     *
     * This mirrors the OpenSearch index structure:
     * {
     *   "embedding": [vector],
     *   "authors": [
     *     {"name": "Author 1"},
     *     {"name": "Author 2"}
     *   ]
     * }
     */
    @Test
    public void testMergesWithRootNullVectorAndNestedChildren() throws IOException {
        final Map<String, float[]> docs = new HashMap<>();

        final int dimension = 128;
        final int k = 1;
        final String VECTOR_FIELD = "embedding";
        final String AUTHOR_NAME_FIELD = "authors.name";
        final String DOC_TYPE_FIELD = "type"; // To distinguish parent from child docs

        log.info("Testing merges with root-level nullable vectors and nested child documents");

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec(1));
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, config)) {
            int parentDocId = 0;
            final int parentDocIdWithoutVector = random().nextInt(2);

            // Segment 1: 3 parent documents, each with 1-3 nested child documents
            log.info("Creating segment 1: 2 parent docs with nested children");
            for (int i = 0; i < 3; i++) {
                // Create nested structure: children first, then parent
                int numChildren = 1 + (i % 3); // 1-3 children per parent
                Document[] docBlock = new Document[numChildren + 1];

                // Add child documents (authors)
                for (int c = 0; c < numChildren; c++) {
                    Document childDoc = new Document();
                    childDoc.add(new StringField(DOC_TYPE_FIELD, "nested", Field.Store.YES));
                    childDoc.add(new TextField(AUTHOR_NAME_FIELD, "Author_" + parentDocId + "_" + c, Field.Store.YES));
                    childDoc.add(new StringField("parent_id", String.valueOf(parentDocId), Field.Store.YES));
                    docBlock[c] = childDoc;
                }

                // Add parent document with vector (must be last in block)
                Document parentDoc = new Document();
                // No vector for this document
                if (parentDocId != parentDocIdWithoutVector) {
                    float[] vector = new float[dimension];
                    Arrays.fill(vector, (parentDocId + 1) * random().nextFloat(1.0f));
                    parentDoc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, VectorSimilarityFunction.COSINE));
                    docs.put(Integer.toString(parentDocId), vector);
                }
                parentDoc.add(new StringField(TEST_ID_FIELD, String.valueOf(parentDocId), Field.Store.YES));
                parentDoc.add(new StringField(DOC_TYPE_FIELD, "parent", Field.Store.YES));
                parentDoc.add(new NumericDocValuesField("num_children", numChildren));
                docBlock[numChildren] = parentDoc;

                // Add the entire block (children + parent) atomically
                writer.addDocuments(Arrays.asList(docBlock));
                parentDocId++;
            }
            writer.commit();

            // Delete 1 parent documents (and their children) from segment 1
            log.info("Deleting 1 parent document 2 from segment 1");
            writer.deleteDocuments(new Term(TEST_ID_FIELD, String.valueOf(2)));
            writer.commit();

            // First intermediate merge
            log.info("Performing first intermediate merge");
            writer.forceMerge(1);

            // Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after final merge", 1, reader.getContext().leaves().size());

                // Count parent documents
                IndexSearcher searcher = newSearcher(reader);
                TopDocs parentDocs = searcher.search(new TermQuery(new Term(DOC_TYPE_FIELD, "parent")), Integer.MAX_VALUE);
                Assert.assertEquals("Should have correct number of parent docs", 2, parentDocs.totalHits.value());

                // Verify vector search works on parent documents
                log.info("Verifying vector search on merged index with nested structure");
                final float[] target = new float[dimension];
                Arrays.fill(target, 0.5f);
                JVectorKnnFloatVectorQuery query = getJVectorKnnFloatVectorQuery(
                    VECTOR_FIELD,
                    target,
                    k,
                    new TermQuery(new Term(DOC_TYPE_FIELD, "parent")) // Only search parent docs
                );
                TopDocs topDocs = searcher.search(query, k);
                Assert.assertEquals("Should return k results", k, topDocs.totalHits.value());

                // Verify results are parent documents with vectors
                log.info("Verifying search results are parent documents");
                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String docType = doc.get(DOC_TYPE_FIELD);
                    String id = doc.get(TEST_ID_FIELD);

                    Assert.assertEquals("Result should be a parent document", "parent", docType);
                    assertNotNull("Parent document should have ID", id);

                    log.info("Result {}: parent doc ID = {}, score = {}", i, id, topDocs.scoreDocs[i].score);
                }

                topDocs = searcher.search(new MatchAllDocsQuery(), 8);
                Assert.assertEquals("Should return k results", 8, topDocs.totalHits.value());

                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String id = doc.get(TEST_ID_FIELD);
                    assertThat(getVector(reader, VECTOR_FIELD, topDocs.scoreDocs[i].doc), equalTo(docs.get(id)));
                    log.info("Result {}: doc ID = {}", i, id);
                }

                topDocs = searcher.search(new TermQuery(new Term(DOC_TYPE_FIELD, "parent")), 10);
                Assert.assertEquals("Should return k results", 2, topDocs.totalHits.value());

                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String id = doc.get(TEST_ID_FIELD);
                    assertThat(getVector(reader, VECTOR_FIELD, topDocs.scoreDocs[i].doc), equalTo(docs.get(id)));
                    log.info("Result {}: doc ID = {}", i, id);
                }
            }
        }
    }

    /**
     * Test merges with root-level vector field and nested child documents,
     * one of the documents has root-level vector field set to null, all child
     * documents have vectors.
     *
     * This test validates the scenario where:
     * - Vector field ("embedding") is at the root (parent) document level
     * - Nested child documents ("authors") contain metadata without vectors
     * - Documents are merged across multiple segments with deletions
     *
     * In Lucene, nested documents are stored as:
     * - Child documents come first (authors)
     * - Parent document comes last (with vector)
     * - All are added as a single block using addDocuments()
     *
     * This mirrors the OpenSearch index structure:
     * {
     *   "embedding": [vector],
     *   "authors": [
     *     {"name": "Author 1"},
     *     {"name": "Author 2"}
     *   ]
     * }
     */
    @Test
    public void testMergesWithRootVectorAndNestedChildrenVectors() throws IOException {
        final Map<String, float[]> docs = new HashMap<>();

        final int dimension = 128;
        final int k = 1;
        final String VECTOR_FIELD = "embedding";
        final String AUTHOR_NAME_FIELD = "authors.name";
        final String DOC_TYPE_FIELD = "type"; // To distinguish parent from child docs

        log.info("Testing merges with root-level nullable vectors and nested child documents");

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec(1));
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, config)) {
            int parentDocId = 0;
            final int parentDocIdWithoutVector = random().nextInt(2);

            // Segment 1: 3 parent documents, each with 1-3 nested child documents
            log.info("Creating segment 1: 2 parent docs with nested children");
            for (int i = 0; i < 3; i++) {
                // Create nested structure: children first, then parent
                int numChildren = 1 + (i % 3); // 1-3 children per parent
                Document[] docBlock = new Document[numChildren + 1];

                // Add child documents (authors)
                for (int c = 0; c < numChildren; c++) {
                    Document childDoc = new Document();
                    float[] vector = new float[dimension];
                    Arrays.fill(vector, (c + 1) * random().nextFloat(1.0f));
                    final String childDocId = String.valueOf(parentDocId) + "-" + String.valueOf(c);
                    childDoc.add(new StringField(DOC_TYPE_FIELD, "nested", Field.Store.YES));
                    childDoc.add(new TextField(AUTHOR_NAME_FIELD, "Author_" + parentDocId + "_" + c, Field.Store.YES));
                    childDoc.add(new StringField("parent_id", String.valueOf(parentDocId), Field.Store.YES));
                    childDoc.add(new StringField(TEST_ID_FIELD, childDocId, Field.Store.YES));
                    childDoc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, VectorSimilarityFunction.COSINE));
                    docs.put(childDocId, vector);
                    docBlock[c] = childDoc;
                }

                // Add parent document with vector (must be last in block)
                Document parentDoc = new Document();
                // No vector for this document
                if (parentDocId != parentDocIdWithoutVector) {
                    float[] vector = new float[dimension];
                    Arrays.fill(vector, (parentDocId + 1) * random().nextFloat(1.0f));
                    parentDoc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, VectorSimilarityFunction.COSINE));
                    docs.put(Integer.toString(parentDocId), vector);
                }
                parentDoc.add(new StringField(TEST_ID_FIELD, String.valueOf(parentDocId), Field.Store.YES));
                parentDoc.add(new StringField(DOC_TYPE_FIELD, "parent", Field.Store.YES));
                parentDoc.add(new NumericDocValuesField("num_children", numChildren));
                docBlock[numChildren] = parentDoc;

                // Add the entire block (children + parent) atomically
                writer.addDocuments(Arrays.asList(docBlock));
                parentDocId++;
            }
            writer.commit();

            // Delete 1 parent documents (and their children) from segment 1
            log.info("Deleting 1 parent document 2 from segment 1");
            writer.deleteDocuments(new Term(TEST_ID_FIELD, String.valueOf(2)));
            writer.commit();

            // First intermediate merge
            log.info("Performing first intermediate merge");
            writer.forceMerge(1);

            // Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after final merge", 1, reader.getContext().leaves().size());

                // Count parent documents
                IndexSearcher searcher = newSearcher(reader);
                TopDocs parentDocs = searcher.search(new TermQuery(new Term(DOC_TYPE_FIELD, "parent")), Integer.MAX_VALUE);
                Assert.assertEquals("Should have correct number of parent docs", 2, parentDocs.totalHits.value());

                // Verify vector search works on parent documents
                log.info("Verifying vector search on merged index with nested structure");
                final float[] target = new float[dimension];
                Arrays.fill(target, 0.5f);
                JVectorKnnFloatVectorQuery query = getJVectorKnnFloatVectorQuery(
                    VECTOR_FIELD,
                    target,
                    k,
                    new TermQuery(new Term(DOC_TYPE_FIELD, "parent")) // Only search parent docs
                );
                TopDocs topDocs = searcher.search(query, k);
                Assert.assertEquals("Should return k results", k, topDocs.totalHits.value());

                // Verify results are parent documents with vectors
                log.info("Verifying search results are parent documents");
                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String docType = doc.get(DOC_TYPE_FIELD);
                    String id = doc.get(TEST_ID_FIELD);

                    Assert.assertEquals("Result should be a parent document", "parent", docType);
                    assertNotNull("Parent document should have ID", id);

                    log.info("Result {}: parent doc ID = {}, score = {}", i, id, topDocs.scoreDocs[i].score);
                }

                topDocs = searcher.search(new MatchAllDocsQuery(), 8);
                Assert.assertEquals("Should return k results", 8, topDocs.totalHits.value());

                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String id = doc.get(TEST_ID_FIELD);
                    assertThat(getVector(reader, VECTOR_FIELD, topDocs.scoreDocs[i].doc), equalTo(docs.get(id)));
                    log.info("Result {}: doc ID = {}", i, id);
                }

                topDocs = searcher.search(new TermQuery(new Term(DOC_TYPE_FIELD, "parent")), 10);
                Assert.assertEquals("Should return k results", 2, topDocs.totalHits.value());

                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String id = doc.get(TEST_ID_FIELD);
                    assertThat(getVector(reader, VECTOR_FIELD, topDocs.scoreDocs[i].doc), equalTo(docs.get(id)));
                    log.info("Result {}: doc ID = {}", i, id);
                }
            }
        }
    }

    /**
     * Test merges with root-level vector field and nested child documents,
     * one of the documents has root-level vector field set to null and its all child
     * documents have no vectors.
     *
     * This test validates the scenario where:
     * - Vector field ("embedding") is at the root (parent) document level
     * - Nested child documents ("authors") contain metadata without vectors
     * - Documents are merged across multiple segments with deletions
     *
     * In Lucene, nested documents are stored as:
     * - Child documents come first (authors)
     * - Parent document comes last (with vector)
     * - All are added as a single block using addDocuments()
     *
     * This mirrors the OpenSearch index structure:
     * {
     *   "embedding": [vector],
     *   "authors": [
     *     {"name": "Author 1"},
     *     {"name": "Author 2"}
     *   ]
     * }
     */
    @Test
    public void testMergesWithRootNoVectorAndNestedChildrenNoVector() throws IOException {
        final Map<String, float[]> docs = new HashMap<>();

        final int dimension = 128;
        final int k = 1;
        final String VECTOR_FIELD = "embedding";
        final String AUTHOR_NAME_FIELD = "authors.name";
        final String DOC_TYPE_FIELD = "type"; // To distinguish parent from child docs

        log.info("Testing merges with root-level nullable vectors and nested child documents");

        IndexWriterConfig config = newIndexWriterConfig();
        config.setUseCompoundFile(false);
        config.setCodec(getCodec(1));
        config.setMergePolicy(new ForceMergesOnlyMergePolicy());
        config.setMergeScheduler(new SerialMergeScheduler());

        final Path indexPath = createTempDir();

        try (FSDirectory dir = FSDirectory.open(indexPath); IndexWriter writer = new IndexWriter(dir, config)) {
            int parentDocId = 0;
            final int parentDocIdWithoutVector = random().nextInt(2);

            // Segment 1: 3 parent documents, each with 1-3 nested child documents
            log.info("Creating segment 1: 2 parent docs with nested children");
            for (int i = 0; i < 3; i++) {
                // Create nested structure: children first, then parent
                int numChildren = 1 + (i % 3); // 1-3 children per parent
                Document[] docBlock = new Document[numChildren + 1];

                // Add child documents (authors)
                for (int c = 0; c < numChildren; c++) {
                    Document childDoc = new Document();
                    final String childDocId = String.valueOf(parentDocId) + "-" + String.valueOf(c);
                    childDoc.add(new StringField(DOC_TYPE_FIELD, "nested", Field.Store.YES));
                    childDoc.add(new TextField(AUTHOR_NAME_FIELD, "Author_" + parentDocId + "_" + c, Field.Store.YES));
                    childDoc.add(new StringField("parent_id", String.valueOf(parentDocId), Field.Store.YES));
                    childDoc.add(new StringField(TEST_ID_FIELD, childDocId, Field.Store.YES));
                    if (parentDocId != parentDocIdWithoutVector) {
                        float[] vector = new float[dimension];
                        Arrays.fill(vector, (c + 1) * random().nextFloat(1.0f));
                        childDoc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, VectorSimilarityFunction.COSINE));
                        docs.put(childDocId, vector);
                    }
                    docBlock[c] = childDoc;
                }

                // Add parent document with vector (must be last in block)
                Document parentDoc = new Document();
                // No vector for this document
                if (parentDocId != parentDocIdWithoutVector) {
                    float[] vector = new float[dimension];
                    Arrays.fill(vector, (parentDocId + 1) * random().nextFloat(1.0f));
                    parentDoc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, VectorSimilarityFunction.COSINE));
                    docs.put(Integer.toString(parentDocId), vector);
                }
                parentDoc.add(new StringField(TEST_ID_FIELD, String.valueOf(parentDocId), Field.Store.YES));
                parentDoc.add(new StringField(DOC_TYPE_FIELD, "parent", Field.Store.YES));
                parentDoc.add(new NumericDocValuesField("num_children", numChildren));
                docBlock[numChildren] = parentDoc;

                // Add the entire block (children + parent) atomically
                writer.addDocuments(Arrays.asList(docBlock));
                parentDocId++;
            }
            writer.commit();

            // Delete 1 parent documents (and their children) from segment 1
            log.info("Deleting 1 parent document 2 from segment 1");
            writer.deleteDocuments(new Term(TEST_ID_FIELD, String.valueOf(2)));
            writer.commit();

            // First intermediate merge
            log.info("Performing first intermediate merge");
            writer.forceMerge(1);

            // Verify the merged index
            try (IndexReader reader = DirectoryReader.open(writer)) {
                Assert.assertEquals("Should have 1 segment after final merge", 1, reader.getContext().leaves().size());

                // Count parent documents
                IndexSearcher searcher = newSearcher(reader);
                TopDocs parentDocs = searcher.search(new TermQuery(new Term(DOC_TYPE_FIELD, "parent")), Integer.MAX_VALUE);
                Assert.assertEquals("Should have correct number of parent docs", 2, parentDocs.totalHits.value());

                // Verify vector search works on parent documents
                log.info("Verifying vector search on merged index with nested structure");
                final float[] target = new float[dimension];
                Arrays.fill(target, 0.5f);
                JVectorKnnFloatVectorQuery query = getJVectorKnnFloatVectorQuery(
                    VECTOR_FIELD,
                    target,
                    k,
                    new TermQuery(new Term(DOC_TYPE_FIELD, "parent")) // Only search parent docs
                );
                TopDocs topDocs = searcher.search(query, k);
                Assert.assertEquals("Should return k results", k, topDocs.totalHits.value());

                // Verify results are parent documents with vectors
                log.info("Verifying search results are parent documents");
                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String docType = doc.get(DOC_TYPE_FIELD);
                    String id = doc.get(TEST_ID_FIELD);

                    Assert.assertEquals("Result should be a parent document", "parent", docType);
                    assertNotNull("Parent document should have ID", id);

                    log.info("Result {}: parent doc ID = {}, score = {}", i, id, topDocs.scoreDocs[i].score);
                }

                topDocs = searcher.search(new MatchAllDocsQuery(), 8);
                Assert.assertEquals("Should return k results", 8, topDocs.totalHits.value());

                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String id = doc.get(TEST_ID_FIELD);
                    assertThat(getVector(reader, VECTOR_FIELD, topDocs.scoreDocs[i].doc), equalTo(docs.get(id)));
                    log.info("Result {}: doc ID = {}", i, id);
                }

                topDocs = searcher.search(new TermQuery(new Term(DOC_TYPE_FIELD, "parent")), 10);
                Assert.assertEquals("Should return k results", 2, topDocs.totalHits.value());

                for (int i = 0; i < topDocs.scoreDocs.length; i++) {
                    Document doc = reader.storedFields().document(topDocs.scoreDocs[i].doc);
                    String id = doc.get(TEST_ID_FIELD);
                    assertThat(getVector(reader, VECTOR_FIELD, topDocs.scoreDocs[i].doc), equalTo(docs.get(id)));
                    log.info("Result {}: doc ID = {}", i, id);
                }
            }
        }
    }

    private static float[] getVector(final IndexReader reader, final String field, final int doc) throws IOException {
        for (LeafReaderContext context : reader.leaves()) {
            final FloatVectorValues vectorValues = context.reader().getFloatVectorValues(field);
            if (vectorValues != null) {
                final var docIdSetIterator = vectorValues.iterator(); // iterator for all the vectors with values
                int docId = -1;
                for (docId = docIdSetIterator.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = docIdSetIterator.nextDoc()) {
                    if (docId == doc) {
                        final int index = docIdSetIterator.index();
                        if (index == GraphNodeIdToDocMap.NO_VECTOR_OR_DELETED_DOC) {
                            return null; /* no vector value set */
                        } else {
                            return vectorValues.vectorValue(index);
                        }
                    }
                }
            }
        }

        throw new IllegalStateException("The docId " + doc + " expected but was not found");
    }
}

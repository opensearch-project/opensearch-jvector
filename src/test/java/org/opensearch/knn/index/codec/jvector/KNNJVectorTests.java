/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FSLockFactory;
import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.opensearch.knn.index.codec.jvector.JVectorFormat.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION;

/**
 * Test used specifically for JVector
 */
// Currently {@link IndexGraphBuilder} is using the default ForkJoinPool.commonPool() which is not being shutdown.
// Ignore thread leaks until we remove the ForkJoinPool.commonPool() usage from IndexGraphBuilder
// TODO: Wire the execution thread pool to {@link IndexGraphBuilder} to avoid the failure of the UT due to leaked thread pool warning.
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "")
@Log4j2
public class KNNJVectorTests extends LuceneTestCase {

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
        int k = 3; // The number of nearest neighbours to gather
        int totalNumberOfDocs = 10;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        // TODO: re-enable this after fixing the compound file augmentation for JVector
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(new JVectorCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (Directory dir = newFSDirectory(indexPath); RandomIndexWriter w = new RandomIndexWriter(random(), dir, indexWriterConfig)) {
            final float[] target = new float[] { 0.0f, 0.0f };
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = new float[] { 0.0f, 1.0f / i };
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField("test_field", source, VectorSimilarityFunction.EUCLIDEAN));
                w.addDocument(doc);
            }
            log.info("Flushing docs to make them discoverable on the file system");
            w.commit();

            try (IndexReader reader = w.getReader()) {
                log.info("We should now have a single segment with 10 documents");
                Assert.assertEquals(1, reader.getContext().leaves().size());
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
     * No merge.
     */
    @Test
    public void testJVectorKnnIndex_multipleSegments() throws IOException {
        int k = 3; // The number of nearest neighbours to gather
        int totalNumberOfDocs = 10;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        // TODO: re-enable this after fixing the compound file augmentation for JVector
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(new JVectorCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(false));
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (
            FSDirectory dir = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());
            IndexWriter w = new IndexWriter(dir, indexWriterConfig)
        ) {
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
        // TODO: re-enable this after fixing the compound file augmentation for JVector
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(new JVectorCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        indexWriterConfig.setMergeScheduler(new SerialMergeScheduler());
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (
            FSDirectory dir = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());
            IndexWriter w = new IndexWriter(dir, indexWriterConfig)
        ) {
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
                KnnFloatVectorQuery knnFloatVectorQuery = new KnnFloatVectorQuery("test_field", target, k, filterQuery);
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
     * Test to verify that the Lucene codec is able to successfully search for the nearest neighbours
     * in the index.
     * Single field is used to store the vectors.
     * Documents are stored in potentially multiple segments.
     * Multiple commits.
     * Multiple merges.
     */
    @Test
    public void testLuceneKnnIndex_multipleMerges() throws IOException {
        int k = 3; // The number of nearest neighbours to gather
        int totalNumberOfDocs = 10;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        // TODO: re-enable this after fixing the compound file augmentation for JVector
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(new JVectorCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        indexWriterConfig.setMergeScheduler(new SerialMergeScheduler());
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (
            FSDirectory dir = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());
            IndexWriter w = new IndexWriter(dir, indexWriterConfig)
        ) {
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
                KnnFloatVectorQuery knnFloatVectorQuery = new KnnFloatVectorQuery("test_field", target, k, filterQuery);
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
        int k = 3; // The number of nearest neighbours to gather
        int totalNumberOfDocs = 10;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        // TODO: re-enable this after fixing the compound file augmentation for JVector
        indexWriterConfig.setUseCompoundFile(true);
        indexWriterConfig.setCodec(new JVectorCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(true));
        indexWriterConfig.setMergeScheduler(new SerialMergeScheduler());
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (
            FSDirectory dir = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());
            IndexWriter w = new IndexWriter(dir, indexWriterConfig)
        ) {
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
                KnnFloatVectorQuery knnFloatVectorQuery = new KnnFloatVectorQuery("test_field", target, k, filterQuery);
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
        indexWriterConfig.setCodec(new JVectorCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(true));
        indexWriterConfig.setMergeScheduler(new SerialMergeScheduler());
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (
            FSDirectory dir = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());
            IndexWriter w = new IndexWriter(dir, indexWriterConfig)
        ) {
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
                KnnFloatVectorQuery knnFloatVectorQuery = new KnnFloatVectorQuery("test_field", target, k, filterQuery);
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
     * TODO: Create Product Quantization support for JVector codec
     */
    @Test
    public void testJVectorKnnIndex_simpleCase_withBinaryVector() throws IOException {
        int k = 3; // The number of nearest neighbours to gather
        int totalNumberOfDocs = 10;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        // TODO: re-enable this after fixing the compound file augmentation for JVector
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(new JVectorCodec());
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
        // TODO: re-enable this after fixing the compound file augmentation for JVector
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(new JVectorCodec());
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (Directory dir = newFSDirectory(indexPath); RandomIndexWriter w = new RandomIndexWriter(random(), dir, indexWriterConfig)) {
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

            try (IndexReader reader = w.getReader()) {
                log.info("Applying filter to the KNN search");
                final Query filterQuery = new TermQuery(new Term("filter_field", "even"));
                final IndexSearcher searcher = newSearcher(reader);
                KnnFloatVectorQuery knnFloatVectorQuery = new KnnFloatVectorQuery("test_field", target, k, filterQuery);
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
        int totalNumberOfDocs = DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION;
        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        // TODO: re-enable this after fixing the compound file augmentation for JVector
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(new JVectorCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION));
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        // We set the below parameters to make sure no permature flush will occur, this way we can have a single segment, and we can force
        // test the quantization case
        indexWriterConfig.setMaxBufferedDocs(10000); // force flush every 10000 docs, this way we make sure that we only have a single
                                                     // segment for a totalNumberOfDocs < 1000
        indexWriterConfig.setRAMPerThreadHardLimitMB(1000); // 1000MB per thread, this way we make sure that no premature flush will occur
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (
            FSDirectory dir = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());
            IndexWriter w = new IndexWriter(dir, indexWriterConfig)
        ) {
            final float[] target = new float[] {
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f };
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = new float[] {
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    i };
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
                KnnFloatVectorQuery knnFloatVectorQuery = new KnnFloatVectorQuery("test_field", target, k, filterQuery);
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
     * Test the simple case of quantization where we have the perfect batch single batch size each time with a merge of
     * multiple segments
     */
    @Test
    public void testJVectorKnnIndex_happyCase_withQuantization_multipleSegments() throws IOException {
        final int k = 50; // The number of nearest neighbours to gather, we set a high number here to avoid an inaccurate result and
                          // jittery tests
        final int perfectBatchSize = DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION; // MINIMUM_BATCH_SIZE_FOR_QUANTIZATION is the minimal
                                                                                  // batch size that will trigger a quantization without
                                                                                  // breaking it, generally speaking the batch size can't be
                                                                                  // lower than the number of clusters
        final int totalNumberOfDocs = perfectBatchSize * 2;

        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        // TODO: re-enable this after fixing the compound file augmentation for JVector
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(new JVectorCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION));
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        // We set the below parameters to make sure no permature flush will occur, this way we can have a single segment, and we can force
        // test the quantization case
        indexWriterConfig.setMaxBufferedDocs(10000); // force flush every 10000 docs, this way we make sure that we only have a single
        // segment for a totalNumberOfDocs < 1000
        indexWriterConfig.setRAMPerThreadHardLimitMB(1000); // 1000MB per thread, this way we make sure that no premature flush will occur
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (
            FSDirectory dir = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());
            IndexWriter w = new IndexWriter(dir, indexWriterConfig)
        ) {
            final float[] target = new float[] {
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f };
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = new float[] {
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    i };
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField("test_field", source, VectorSimilarityFunction.EUCLIDEAN));
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
                KnnFloatVectorQuery knnFloatVectorQuery = new KnnFloatVectorQuery("test_field", target, k, filterQuery);
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
     */
    @Test
    public void testJVectorKnnIndex_mixedBatchSizes_withQuantization_multipleMerges() throws IOException {
        final int k = 50; // The number of nearest neighbours to gather, we set a high number here to avoid an inaccurate result and
                          // jittery tests
        final int notIdealBatchSize = DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION / 3; // Batch size that is not ideal for quantization and
                                                                                       // shouldn't trigger it
        final int totalNumberOfDocs = notIdealBatchSize * 3; // 3 batches of documents each will result in quantization only when the merge
                                                             // is triggered, and we have a batch size of {@link
                                                             // MINIMUM_BATCH_SIZE_FOR_QUANTIZATION} as a result of merging all the smaller
                                                             // batches

        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        // TODO: re-enable this after fixing the compound file augmentation for JVector
        indexWriterConfig.setUseCompoundFile(false);
        indexWriterConfig.setCodec(new JVectorCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION));
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy());
        // We set the below parameters to make sure no permature flush will occur, this way we can have a single segment, and we can force
        // test the quantization case
        indexWriterConfig.setMaxBufferedDocs(10000); // force flush every 10000 docs, this way we make sure that we only have a single
        // segment for a totalNumberOfDocs < 1000
        indexWriterConfig.setRAMPerThreadHardLimitMB(1000); // 1000MB per thread, this way we make sure that no premature flush will occur
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (
            FSDirectory dir = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());
            IndexWriter w = new IndexWriter(dir, indexWriterConfig)
        ) {
            final float[] target = new float[] {
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f };
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = new float[] {
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    i };
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
                KnnFloatVectorQuery knnFloatVectorQuery = new KnnFloatVectorQuery("test_field", target, k, filterQuery);
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
        final int notIdealBatchSize = DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION / 3; // Batch size that is not ideal for quantization and
        // shouldn't trigger it
        final int totalNumberOfDocs = notIdealBatchSize * 30; // 3 batches of documents each will result in quantization only when the merge
        // is triggered, and we have a batch size of {@link
        // MINIMUM_BATCH_SIZE_FOR_QUANTIZATION} as a result of merging all the smaller
        // batches

        IndexWriterConfig indexWriterConfig = LuceneTestCase.newIndexWriterConfig();
        // TODO: re-enable this after fixing the compound file augmentation for JVector
        indexWriterConfig.setUseCompoundFile(true);
        indexWriterConfig.setCodec(new JVectorCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION));
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(true));
        // We set the below parameters to make sure no permature flush will occur, this way we can have a single segment, and we can force
        // test the quantization case
        indexWriterConfig.setMaxBufferedDocs(10000); // force flush every 10000 docs, this way we make sure that we only have a single
        // segment for a totalNumberOfDocs < 1000
        indexWriterConfig.setRAMPerThreadHardLimitMB(1000); // 1000MB per thread, this way we make sure that no premature flush will occur
        final Path indexPath = createTempDir();
        log.info("Index path: {}", indexPath);
        try (
            FSDirectory dir = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());
            IndexWriter w = new IndexWriter(dir, indexWriterConfig)
        ) {
            final float[] target = new float[] {
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f,
                0.0f };
            for (int i = 1; i < totalNumberOfDocs + 1; i++) {
                final float[] source = new float[] {
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    0.0f,
                    i };
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
                KnnFloatVectorQuery knnFloatVectorQuery = new KnnFloatVectorQuery("test_field", target, k, filterQuery);
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
        return ((float) totalRelevantDocs) / ((float) topDocs.scoreDocs.length);
    }

}

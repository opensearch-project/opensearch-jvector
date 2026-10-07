/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;
import static org.opensearch.knn.index.engine.CommonTestUtils.getCodec;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;

@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "")
public class JVectorWriterAbortTests extends LuceneTestCase {

    private static final String VECTOR_FIELD = "test_vector";
    private static final int DIMENSION = 64;

    private void createSegment(IndexWriter writer, int numDocs) throws IOException {
        for (int i = 0; i < numDocs; i++) {
            Document doc = new Document();
            float[] vector = new float[DIMENSION];
            for (int d = 0; d < DIMENSION; d++) {
                vector[d] = random().nextFloat();
            }
            doc.add(new KnnFloatVectorField(VECTOR_FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
            doc.add(new StringField("id", "doc_" + i, Store.NO));
            writer.addDocument(doc);
        }
        writer.commit();
    }

    private void runAbortTest(Codec codec, int numSegments, int docsPerSegment) throws Exception {
        try (Directory dir = newDirectory()) {
            IndexWriterConfig iwc = newIndexWriterConfig();
            iwc.setCodec(codec);

            IndexWriter writer = new IndexWriter(dir, iwc);

            // Create segments to build up a noticeable merge workload
            for (int s = 0; s < numSegments; s++) {
                createSegment(writer, docsPerSegment);
            }

            // Start forceMerge in background
            CountDownLatch mergeStartedLatch = new CountDownLatch(1);
            AtomicReference<Throwable> mergeError = new AtomicReference<>();

            Thread mergeThread = new Thread(() -> {
                try {
                    mergeStartedLatch.countDown();
                    writer.forceMerge(1);
                } catch (Throwable t) {
                    mergeError.set(t);
                }
            });
            mergeThread.start();

            // Wait until the merge thread has kicked off
            assertTrue(mergeStartedLatch.await(5, TimeUnit.SECONDS));
            // Give brief moment for worker threads to start graph construction
            Thread.sleep(50);

            // Rollback the writer — internally calls IndexWriter#abortMerges(),
            long startRollback = System.currentTimeMillis();
            writer.rollback();
            mergeThread.join(10000); // 10 second timeout

            long rollbackElapsed = System.currentTimeMillis() - startRollback;

            assertFalse("Merge thread hung and did not abort!", mergeThread.isAlive());
            assertTrue("Rollback took too long (" + rollbackElapsed + "ms), abort was delayed", rollbackElapsed < 5000);

            // Verify that the merge was aborted with an IOException
            assertNotNull("Merge should have thrown an exception on abort", mergeError.get());
            assertTrue("Expected IOException on abort but got: " + mergeError.get(), mergeError.get() instanceof IOException);
        }
    }

    /**
     * Tests aborting during full graph rebuild from scratch (getGraph).
     */
    @Test
    public void testAbortDuringScratchGraphBuild() throws Exception {
        // leadingSegmentMergeDisabled = true forces getGraph() scratch build
        Codec codec = getCodec(Integer.MAX_VALUE, true, false);
        runAbortTest(codec, 4, 3000);
    }

    /**
     * Tests aborting during leading segment incremental merge (tryLeadingSegmentMerge).
     */
    @Test
    public void testAbortDuringLeadingSegmentMerge() throws Exception {
        // leadingSegmentMergeDisabled = false allows tryLeadingSegmentMerge()
        Codec codec = getCodec(Integer.MAX_VALUE, false, false);
        runAbortTest(codec, 4, 3000);
    }
}

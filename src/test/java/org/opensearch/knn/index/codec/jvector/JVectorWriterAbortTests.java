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
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;
import static org.opensearch.knn.index.engine.CommonTestUtils.getCodec;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;

@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "")
public class JVectorWriterAbortTests extends LuceneTestCase {

    private static final String VECTOR_FIELD = "test_vector";
    private static final int DIMENSION = 64;
    private static final int MERGE_ABORT_CHECK_NUM_ORDINALS = 1000;

    // -----------------------------------------------------------------------
    // checkMergeAborted
    // -----------------------------------------------------------------------

    /** Before the first interval boundary (ord=0..998), checkAborted must never be called. */
    @Test
    public void testCheckMergeAbortedDoesNotFireBeforeInterval() throws Exception {
        MergeState mergeState = mock(MergeState.class);
        // (ord+1) % 1000 == 0 first fires at ord=999; ords 0..998 must never trigger it
        for (int ord = 0; ord < MERGE_ABORT_CHECK_NUM_ORDINALS - 1; ord++) {
            JVectorWriter.checkMergeAborted(mergeState, ord);
        }
        verify(mergeState, never()).checkAborted();
    }

    /** checkAborted fires at ord=999 (1000th node) and ord=1999 (2000th node), nowhere else. */
    @Test
    public void testCheckMergeAbortedFiresOnlyAtIntervalMultiples() throws Exception {
        MergeState mergeState = mock(MergeState.class);
        JVectorWriter.checkMergeAborted(mergeState, 0);                                        // does not fire (ord+1=1)
        JVectorWriter.checkMergeAborted(mergeState, MERGE_ABORT_CHECK_NUM_ORDINALS - 1);       // fires (ord+1=1000)
        JVectorWriter.checkMergeAborted(mergeState, MERGE_ABORT_CHECK_NUM_ORDINALS);           // does not fire (ord+1=1001)
        JVectorWriter.checkMergeAborted(mergeState, 2 * MERGE_ABORT_CHECK_NUM_ORDINALS - 1);   // fires (ord+1=2000)
        verify(mergeState, times(2)).checkAborted();
    }

    /** Abort triggered before an interval boundary is not detected — check hasn't fired yet. */
    @Test
    public void testCheckMergeAbortedDoesNotThrowBeforeInterval() throws Exception {
        MergeState mergeState = mock(MergeState.class);
        doThrow(new MergePolicy.MergeAbortedException("aborted")).when(mergeState).checkAborted();

        // ord=1 is before the interval — no exception expected
        JVectorWriter.checkMergeAborted(mergeState, 1);
    }

    // -----------------------------------------------------------------------
    // Abort during live graph construction
    // -----------------------------------------------------------------------

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

    private void runAbortTest(Codec codec, int numSegments, int docsPerSegment, long maxRollbackMs) throws Exception {
        runAbortTest(codec, numSegments, docsPerSegment, maxRollbackMs, 0);
    }

    private void runAbortTest(Codec codec, int numSegments, int docsPerSegment, long maxRollbackMs, int leadingDocsCount) throws Exception {
        try (Directory dir = newDirectory()) {
            IndexWriterConfig indexWriterConfig = new IndexWriterConfig();
            indexWriterConfig.setCodec(codec);

            IndexWriter writer = new IndexWriter(dir, indexWriterConfig);

            // Optionally flush a dedicated leading segment first so it has an on-disk graph
            // (score cache) before the other segments are written. Required for the
            // tryLeadingSegmentMerge() path; skipped when leadingDocsCount == 0.
            if (leadingDocsCount > 0) {
                createSegment(writer, leadingDocsCount);
            }

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

            assertFalse("Merge thread hung and did not abort", mergeThread.isAlive());
            assertTrue(
                "Rollback took too long (" + rollbackElapsed + "ms, limit=" + maxRollbackMs + "ms)",
                rollbackElapsed < maxRollbackMs
            );
            // Verify that the merge was aborted with an IOException
            assertNotNull("Merge should have thrown an exception on abort", mergeError.get());
            assertTrue("Expected IOException on abort but got: " + mergeError.get(), mergeError.get() instanceof IOException);
        }
    }

    /** 4 × 3000 = 12 000 vectors > threshold (5000): checkMergeAborted() fires every 1000 nodes inside getGraph(). */
    @Test
    public void testAbortDuringScratchGraphBuild() throws Exception {
        Codec codec = getCodec(Integer.MAX_VALUE, true, false);
        runAbortTest(codec, 4, 3000, 5000);
    }

    @Test
    public void testAbortDuringLeadingSegmentMerge() throws Exception {
        Codec codec = getCodec(Integer.MAX_VALUE, false, false);
        // leadingDocsCount=2000: flushed first so getNeighborsScoreCacheForField finds a graph.
        // 3 × 500 = 1500 non-leading vectors (> 1000) guarantee the abort check fires inside
        // tryLeadingSegmentMerge(). Total 3500 < 5000 keeps getGraph()'s abort path unreachable.
        runAbortTest(codec, 3, 500, 5000, 2000);
    }

    /** flush passes null mergeState — no abort check, graph completes normally. */
    @Test
    public void testFlushPathCompletesWithoutAbortCheck() throws Exception {
        Codec codec = getCodec(Integer.MAX_VALUE, true, false);
        try (Directory dir = newDirectory()) {
            IndexWriterConfig cfg = new IndexWriterConfig();
            cfg.setCodec(codec);
            try (IndexWriter writer = new IndexWriter(dir, cfg)) {
                createSegment(writer, 200);
            }
            assertTrue("Segment files must exist after flush", dir.listAll().length > 0);
        }
    }

    /** 3 × 1000 = 3000 vectors <= threshold (5000) - no abort check */
    @Test
    public void testSmallGraphMergeCompletesWithoutAbortCheck() throws Exception {
        Codec codec = getCodec(Integer.MAX_VALUE, true, false);
        runAbortTest(codec, 3, 1000, 5000);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.StringHelper;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.opensearch.common.util.io.IOUtils;
import org.opensearch.knn.index.codec.jvector.JVectorSegmentQuantizationCache.CloseableLoadedState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.hamcrest.CoreMatchers.equalTo;

/**
 * Tests for {@link JVectorSegmentQuantizationCache}.
 *
 * <p>All tests use {@code quantizationType = QUANTIZATION_TYPE_NONE} (value {@code 0}) with
 * {@code compressedVectorsLength = 0}.  In that combination
 * {@link JVectorIndexQuantization#loadQuantizationState} takes the fast path that returns
 * {@code new LoadedState(null, null)} immediately, without touching {@code index} or
 * {@code directory}.  This lets every test run entirely in-memory, with no mocking framework
 * and no on-disk data.
 */
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "")
public class JVectorSegmentQuantizationCacheTests extends LuceneTestCase {
    private static final String VECTOR_INDEX_FIELD_DATA_FILENAME = "_3_JVectorFormat_0_test_field.data-jvector";

    private static final long OFFSET = 0L;
    private static final long LENGTH = 0L;          // triggers the no-I/O fast path
    private static final long INDEX_OFFSET = 0L;

    private JVectorSegmentQuantizationCache cache;

    @Before
    @Override
    public void setUp() throws Exception {
        super.setUp();
        cache = new JVectorSegmentQuantizationCache();
    }

    @After
    @Override
    public void tearDown() throws Exception {
        super.tearDown();
        assertThat(cache.size(), equalTo(0));
    }

    @Test
    public void testLoadReturnsNonNullHandle() throws IOException {
        try (CloseableLoadedState cls = load(StringHelper.randomId())) {
            assertNotNull("load() must never return null", cls);
            assertEquals("ref count must be 1 right after the first load", 1, cls.state().getRefCount());
            assertNull("nvqInlineQuantization() must be null for QUANTIZATION_TYPE_NONE", cls.nvqInlineQuantization());
            assertNull("pqVectors() must be null for QUANTIZATION_TYPE_NONE", cls.pqVectors());
        }
    }

    @Test
    public void testSecondLoadForSameKeySharesRefCountWrapper() throws IOException {
        final byte[] segId = StringHelper.randomId();
        try (CloseableLoadedState first = load(segId); CloseableLoadedState second = load(segId)) {
            assertSame("two loads for the same key must share the same RefCount instance", first.state(), second.state());
        }
    }

    @Test
    public void testDifferentSegmentIdsYieldIndependentEntries() throws IOException {
        try (CloseableLoadedState a = load(StringHelper.randomId()); CloseableLoadedState b = load(StringHelper.randomId())) {
            assertNotSame("different segment IDs must produce independent cache entries", a.state(), b.state());
        }
    }

    @Test
    public void testDifferentFieldNamesYieldIndependentEntries() throws IOException {
        final byte[] segId = StringHelper.randomId();
        try (
            CloseableLoadedState a = load(segId, "_2_JVectorFormat_0_test_field.data-jvector");
            CloseableLoadedState b = load(segId, "_3_JVectorFormat_0_test_field.data-jvector")
        ) {
            assertNotSame("different field file names must produce independent cache entries", a.state(), b.state());
        }
    }

    @Test
    public void testDifferentOffsetsYieldIndependentEntries() throws IOException {
        final byte[] segId = StringHelper.randomId();

        try (
            CloseableLoadedState a = cache.load(
                JVectorIndexQuantization.QUANTIZATION_TYPE_NONE,
                null,
                null,
                segId,
                VECTOR_INDEX_FIELD_DATA_FILENAME,
                100L,
                LENGTH,
                INDEX_OFFSET
            );
            CloseableLoadedState b = cache.load(
                JVectorIndexQuantization.QUANTIZATION_TYPE_NONE,
                null,
                null,
                segId,
                VECTOR_INDEX_FIELD_DATA_FILENAME,
                200L,
                LENGTH,
                INDEX_OFFSET
            )
        ) {

            assertNotSame("different compressedVectorsOffset values must produce independent entries", a.state(), b.state());
        }
    }

    @Test
    public void testDifferentIndexOffsetsYieldIndependentEntries() throws IOException {
        final byte[] segId = StringHelper.randomId();
        try (
            CloseableLoadedState a = cache.load(
                JVectorIndexQuantization.QUANTIZATION_TYPE_NONE,
                null,
                null,
                segId,
                VECTOR_INDEX_FIELD_DATA_FILENAME,
                OFFSET,
                LENGTH,
                0L
            );
            CloseableLoadedState b = cache.load(
                JVectorIndexQuantization.QUANTIZATION_TYPE_NONE,
                null,
                null,
                segId,
                VECTOR_INDEX_FIELD_DATA_FILENAME,
                OFFSET,
                LENGTH,
                77L
            )
        ) {
            assertNotSame("different vectorIndexOffset values must produce independent entries", a.state(), b.state());
        }
    }

    @Test
    public void testDifferentQuantizationTypesYieldIndependentEntries() throws IOException {
        final byte[] segId = StringHelper.randomId();
        try (
            CloseableLoadedState a = cache.load(
                JVectorIndexQuantization.QUANTIZATION_TYPE_NONE,
                null,
                null,
                segId,
                VECTOR_INDEX_FIELD_DATA_FILENAME,
                OFFSET,
                LENGTH,
                INDEX_OFFSET
            );
            CloseableLoadedState b = cache.load(
                JVectorIndexQuantization.QUANTIZATION_TYPE_PQ,
                null,
                null,
                segId,
                VECTOR_INDEX_FIELD_DATA_FILENAME,
                OFFSET,
                LENGTH,
                INDEX_OFFSET
            )
        ) {
            assertNotSame("different quantization types must produce independent cache entries", a.state(), b.state());
        }
    }

    @Test
    public void testEntryEvictedAfterAllHandlesClosed() throws IOException {
        final byte[] segId = StringHelper.randomId();

        final CloseableLoadedState first = load(segId);
        var firstRef = first.state();
        first.close(); // ref-count → 0, entry removed from map

        // A subsequent load must create a brand-new RefCount entry.
        try (CloseableLoadedState second = load(segId)) {
            assertNotSame("a new load after eviction must produce a fresh RefCount entry", firstRef, second.state());
        }
    }

    @Test
    public void testEntryNotEvictedWhileAtLeastOneHandleIsOpen() throws IOException {
        final byte[] segId = StringHelper.randomId();

        final CloseableLoadedState a = load(segId);
        try (CloseableLoadedState b = load(segId)) {
            var ref = a.state();
            a.close(); // count drops to 1, entry must still be alive

            // A new load while b is still open must reuse the existing entry.
            try (CloseableLoadedState c = load(segId)) {
                assertSame("entry must survive as long as one handle remains open", ref, c.state());
            }
        }
    }

    @Test
    public void testCloseWhenRefCountAlreadyZeroDoesNotThrow() throws IOException {
        final byte[] segId = StringHelper.randomId();

        final CloseableLoadedState cls = load(segId);
        cls.close();
        cls.close();

        assertEquals("ref count must remain 0 after redundant close", 0, cls.state().getRefCount());
    }

    @Test
    public void testConcurrentLoadsForSameKeyShareOneRefCountEntry() throws Exception {
        final int threads = 8;
        final byte[] segId = StringHelper.randomId();

        final CyclicBarrier barrier = new CyclicBarrier(threads);
        final ExecutorService pool = Executors.newFixedThreadPool(threads);
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        final List<Future<CloseableLoadedState>> futures = new ArrayList<>(threads);

        for (int t = 0; t < threads; t++) {
            futures.add(pool.submit(() -> {
                try {
                    barrier.await(10, TimeUnit.SECONDS);
                    return load(segId);
                } catch (Throwable ex) {
                    failure.compareAndSet(null, ex);
                    return null;
                }
            }));
        }

        pool.shutdown();
        assertTrue("thread pool did not terminate in time", pool.awaitTermination(30, TimeUnit.SECONDS));
        assertNull("a worker thread threw an exception", failure.get());

        List<CloseableLoadedState> results = new ArrayList<>(threads);
        for (Future<CloseableLoadedState> f : futures) {
            CloseableLoadedState cls = f.get();
            assertNotNull("load() must not return null from any thread", cls);
            results.add(cls);
        }

        // All threads must have received the same RefCount wrapper.
        var sharedRef = results.get(0).state();
        for (CloseableLoadedState cls : results) {
            assertSame("every concurrent load must share one RefCount entry", sharedRef, cls.state());
        }

        // Ref count must equal the number of live handles.
        assertEquals("ref count must equal number of concurrent holders", threads, sharedRef.getRefCount());

        IOUtils.close(results);
        assertEquals("ref count must reach 0 after all concurrent handles are closed", 0, sharedRef.getRefCount());
    }

    @Test
    public void testConcurrentLoadsForDistinctKeysAreIndependent() throws Exception {
        final int threads = 8;
        final CyclicBarrier barrier = new CyclicBarrier(threads);
        final ExecutorService pool = Executors.newFixedThreadPool(threads);
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        final List<Future<CloseableLoadedState>> futures = new ArrayList<>(threads);

        for (int t = 0; t < threads; t++) {
            final int idx = t;
            futures.add(pool.submit(() -> {
                try {
                    barrier.await(10, TimeUnit.SECONDS);
                    // Each thread uses a unique segment ID → distinct cache key.
                    return load(StringHelper.randomId(), "_" + idx + "_JVectorFormat_0_test_field.data-jvector");
                } catch (Throwable ex) {
                    failure.compareAndSet(null, ex);
                    return null;
                }
            }));
        }

        pool.shutdown();
        assertTrue("thread pool did not terminate in time", pool.awaitTermination(30, TimeUnit.SECONDS));
        assertNull("a worker thread threw an exception", failure.get());

        final List<CloseableLoadedState> results = new ArrayList<>(threads);
        for (Future<CloseableLoadedState> f : futures) {
            results.add(f.get());
        }

        // Every entry must be independent with ref-count exactly 1.
        for (int i = 0; i < results.size(); i++) {
            assertEquals("isolated key must have ref count 1", 1, results.get(i).state().getRefCount());
            for (int j = i + 1; j < results.size(); j++) {
                assertNotSame("distinct keys must not share a RefCount entry", results.get(i).state(), results.get(j).state());
            }
        }

        IOUtils.close(results);
    }

    @Test
    public void testConcurrentLoadAndCloseDoNotCorruptRefCount() throws Exception {
        final int threads = 8;
        final int iterationsPerThread = 100;
        final byte[] segId = StringHelper.randomId();

        final AtomicReference<Throwable> failure = new AtomicReference<>();
        final ExecutorService pool = Executors.newFixedThreadPool(threads);
        final CyclicBarrier barrier = new CyclicBarrier(threads);

        for (int t = 0; t < threads; t++) {
            pool.submit(() -> {
                try {
                    barrier.await(10, TimeUnit.SECONDS);
                    for (int i = 0; i < iterationsPerThread; i++) {
                        try (CloseableLoadedState cls = load(segId)) {
                            Thread.yield();
                        }
                    }
                } catch (Throwable ex) {
                    failure.compareAndSet(null, ex);
                }
            });
        }

        pool.shutdown();
        assertTrue("thread pool did not terminate in time", pool.awaitTermination(30, TimeUnit.SECONDS));
        assertNull("a worker thread threw an exception", failure.get());

        // After all paired load/close cycles there must be no live handle, so the entry
        // must have been evicted. A fresh load must start at ref-count 1.
        try (CloseableLoadedState last = load(segId)) {
            assertEquals("fresh load after all concurrent cycles must have ref count 1", 1, last.state().getRefCount());
        }
    }

    @Test
    public void testConcurrentLoadsThenSequentialClosesLeaveNoLeak() throws Exception {
        final int threads = 8;
        final byte[] segId = StringHelper.randomId();
        final AtomicInteger acquired = new AtomicInteger();

        final CyclicBarrier barrier = new CyclicBarrier(threads);
        final ExecutorService pool = Executors.newFixedThreadPool(threads);
        final List<Future<CloseableLoadedState>> futures = new ArrayList<>(threads);
        final AtomicReference<Throwable> failure = new AtomicReference<>();

        for (int t = 0; t < threads; t++) {
            futures.add(pool.submit(() -> {
                try {
                    barrier.await(10, TimeUnit.SECONDS);
                    CloseableLoadedState cls = load(segId);
                    acquired.incrementAndGet();
                    return cls;
                } catch (Throwable ex) {
                    failure.compareAndSet(null, ex);
                    return null;
                }
            }));
        }

        pool.shutdown();
        assertTrue("thread pool did not terminate in time", pool.awaitTermination(30, TimeUnit.SECONDS));
        assertNull("a worker thread threw an exception", failure.get());

        final int liveHandles = acquired.get();
        List<CloseableLoadedState> results = new ArrayList<>(liveHandles);
        for (Future<CloseableLoadedState> f : futures) {
            results.add(f.get());
        }

        assertEquals("ref count must equal number of successfully acquired handles", liveHandles, results.get(0).state().getRefCount());

        // Close all sequentially and verify the count drains correctly.
        for (int i = 0; i < results.size(); i++) {
            int expected = liveHandles - i - 1;
            results.get(i).close();
            assertEquals("ref count must be " + expected + " after closing handle " + i, expected, results.get(0).state().getRefCount());
        }
    }

    @Test
    public void testConcurrentLoadsDifferentSegmentsNoDeadlock() throws Exception {
        final int threads = 8;
        final int keysPerThread = 5;
        final CyclicBarrier barrier = new CyclicBarrier(threads);
        final ExecutorService pool = Executors.newFixedThreadPool(threads);
        final AtomicReference<Throwable> failure = new AtomicReference<>();

        for (int t = 0; t < threads; t++) {
            final int base = t * keysPerThread;
            pool.submit(() -> {
                try {
                    barrier.await(10, TimeUnit.SECONDS);
                    final List<CloseableLoadedState> handles = new ArrayList<>();
                    for (int k = 0; k < keysPerThread; k++) {
                        handles.add(load(StringHelper.randomId(), "_" + k + "_JVectorFormat_" + base + "_test_field.data-jvector"));
                    }
                    Thread.yield();
                    IOUtils.close(handles);
                } catch (Throwable ex) {
                    failure.compareAndSet(null, ex);
                }
            });
        }

        pool.shutdown();
        assertTrue("thread pool did not terminate in time", pool.awaitTermination(30, TimeUnit.SECONDS));
        assertNull("a worker thread threw an exception", failure.get());
    }

    private CloseableLoadedState load(byte[] segId, String vectorIndexFieldDataFileName) throws IOException {
        return cache.load(
            JVectorIndexQuantization.QUANTIZATION_TYPE_NONE,
            null,
            null,
            segId,
            vectorIndexFieldDataFileName,
            OFFSET,
            LENGTH,
            INDEX_OFFSET
        );
    }

    private CloseableLoadedState load(byte[] segId) throws IOException {
        return load(segId, VECTOR_INDEX_FIELD_DATA_FILENAME);
    }
}

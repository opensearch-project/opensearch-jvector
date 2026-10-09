/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import java.io.Closeable;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.lucene.store.Directory;
import org.apache.lucene.util.RefCount;
import org.apache.lucene.util.StringHelper;
import org.opensearch.common.CheckedSupplier;
import org.opensearch.knn.index.codec.jvector.JVectorIndexQuantization.LoadedState;
import org.opensearch.knn.index.codec.jvector.JVectorSegmentQuantizationCache.QuantizationSupplier;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import lombok.extern.log4j.Log4j2;

/**
 * IndexQuantizationCache: caches the PQ/NVQ compressed vectors for the same segment + field
 * combination so we don't inflate the heap usage that happen when derived sources are turned on.
 */
@Log4j2
class JVectorSegmentQuantizationCache {
    private final ConcurrentMap<String, QuantizationSupplier> cache;

    /**
     * Closeable wrapper over {@link LoadedState}
     */
    static final class CloseableLoadedState implements Closeable {
        private final RefCount<LoadedState> state;
        private final AtomicBoolean closed = new AtomicBoolean(false);

        CloseableLoadedState(RefCount<LoadedState> state) {
            this.state = state;
        }

        RefCount<LoadedState> state() {
            return state;
        }

        @Override
        public void close() throws IOException {
            if (closed.compareAndSet(false, true)) {
                state.decRef();
            }
        }

        NVQuantization nvqInlineQuantization() {
            return state.get().nvqInlineQuantization();
        }

        PQVectors pqVectors() {
            return state.get().pqVectors();
        }
    }

    /**
     * Lazy thread-safe one-time supplier for quantization state.
     */
    static final class QuantizationSupplier {
        private final String key;
        private final CheckedSupplier<LoadedState, IOException> supplier;
        private volatile RefCount<LoadedState> refCount;

        QuantizationSupplier(String key, CheckedSupplier<LoadedState, IOException> supplier) {
            this.key = key;
            this.supplier = supplier;
        }

        RefCount<LoadedState> get(ConcurrentMap<String, QuantizationSupplier> cache) throws IOException {
            try {
                return acquireAndGet(cache);
            } catch (Throwable t) {
                cache.remove(key, this);
                throw t;
            }
        }

        synchronized private RefCount<LoadedState> acquireAndGet(ConcurrentMap<String, QuantizationSupplier> cache) throws IOException {
            if (this.refCount == null) {
                final LoadedState loaded = supplier.get();
                this.refCount = new RefCount<>(loaded) {
                    @Override
                    protected void release() throws IOException {
                        log.debug("Cleaned cached quantization state for key {}", key);
                        cache.remove(key, QuantizationSupplier.this);
                    }
                };
            } else {
                log.debug("Cached quantization state found for key {}", key);
                this.refCount.incRef();
            }

            return this.refCount;
        }

        boolean isClosed() {
            final RefCount<LoadedState> ref = refCount;
            return ref != null && ref.getRefCount() == 0;
        }

    }

    JVectorSegmentQuantizationCache() {
        cache = new ConcurrentHashMap<>();
    }

    /**
     * Load or retrieve the PQ/NVQ compressed vectors from the cache
     * @param quantizationType quantization type
     * @param index index
     * @param directory directory
     * @param segmentId unique segment id (since same fields could be present in different indices)
     * @param vectorIndexFieldDataFileName vectorIndexFieldDataFileName
     * @param compressedVectorsOffset compressedVectorsOffset
     * @param compressedVectorsLength compressedVectorsLength
     * @param vectorIndexOffset vectorIndexOffset
     * @return PQ/NVQ compressed vectors
     * @throws IOException
     */
    CloseableLoadedState load(
        byte quantizationType,
        OnDiskGraphIndex index,
        Directory directory,
        byte[] segmentId,
        String vectorIndexFieldDataFileName,
        long compressedVectorsOffset,
        long compressedVectorsLength,
        long vectorIndexOffset
    ) throws IOException {
        final String cacheKey = getCacheKey(
            quantizationType,
            segmentId,
            vectorIndexFieldDataFileName,
            compressedVectorsOffset,
            compressedVectorsLength,
            vectorIndexOffset
        );

        final QuantizationSupplier supplier = cache.compute(cacheKey, (key, existing) -> {
            if (existing == null || existing.isClosed()) {
                log.debug("No cached quantization state found for field {}, loaded from disk", vectorIndexFieldDataFileName);
                return new QuantizationSupplier(
                    key,
                    () -> JVectorIndexQuantization.loadQuantizationState(
                        quantizationType,
                        index,
                        directory,
                        vectorIndexFieldDataFileName,
                        compressedVectorsOffset,
                        compressedVectorsLength,
                        vectorIndexOffset
                    )
                );
            }
            return existing;
        });

        final RefCount<LoadedState> state = supplier.get(cache);
        return new CloseableLoadedState(state);
    }

    int size() {
        return cache.size();
    }

    private static String getCacheKey(
        byte quantizationType,
        byte[] segmentId,
        String vectorIndexFieldDataFileName,
        long compressedVectorsOffset,
        long compressedVectorsLength,
        long vectorIndexOffset
    ) {
        return StringHelper.idToString(segmentId)
            + "."
            + vectorIndexFieldDataFileName
            + "."
            + compressedVectorsOffset
            + "."
            + compressedVectorsLength
            + "."
            + vectorIndexOffset
            + "."
            + quantizationType;
    }
}

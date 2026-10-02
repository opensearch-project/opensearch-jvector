/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import java.io.Closeable;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import org.apache.lucene.store.Directory;
import org.apache.lucene.util.RefCount;
import org.apache.lucene.util.StringHelper;
import org.opensearch.knn.index.codec.jvector.JVectorIndexQuantization.LoadedState;

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
    private ConcurrentMap<String, RefCount<LoadedState>> cache;

    /**
     * Closeable wrapper over {@link LoadedState}
     */
    record CloseableLoadedState(RefCount<LoadedState> state) implements Closeable {
        @Override
        public void close() throws IOException {
            if (state.getRefCount() > 0) {
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

        try {
            final RefCount<LoadedState> state = cache.compute(cacheKey, (key, value) -> {
                try {
                    if (value == null || value.getRefCount() == 0) {
                        log.debug("No cached quantization state found for field {}, loaded from disk", vectorIndexFieldDataFileName);
                        return new RefCount<>(
                            JVectorIndexQuantization.loadQuantizationState(
                                quantizationType,
                                index,
                                directory,
                                vectorIndexFieldDataFileName,
                                compressedVectorsOffset,
                                compressedVectorsLength,
                                vectorIndexOffset
                            )
                        ) {
                            protected void release() throws IOException {
                                log.debug("Cleaned cached quantization state for field {}", vectorIndexFieldDataFileName);
                                cache.remove(key, this);
                            }
                        };
                    } else {
                        log.debug("Cached quantization state found for field {}", vectorIndexFieldDataFileName);
                        value.incRef();
                        return value;
                    }
                } catch (IOException ex) {
                    throw new UncheckedIOException(ex);
                }
            });
            return new CloseableLoadedState(state);
        } catch (UncheckedIOException ex) {
            throw ex.getCause();
        }
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

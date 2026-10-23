/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;

/**
 * A {@link ScoreFunction} decorator that issues a prefetch hint to the OS page cache
 * immediately before every vector read, mirroring the pattern established by Lucene's
 * {@code PrefetchableFlatVectorScorer}.
 *
 * <p><b>Motivation.</b> jVector stores its graph (node IDs, inline quantized vectors,
 * and adjacency lists) in a single memory-mapped file opened with {@code MADV_RANDOM},
 * which disables the kernel's sequential read-ahead.  During graph search each call to
 * {@link ScoreFunction#similarityTo(int)} must load the vector for an arbitrary node —
 * a cold-cache miss that stalls the CPU.  By calling
 * {@link OnDiskGraphIndex#prefetchL0Records(int, int)} just before the read we tell the
 * OS ({@code madvise(MADV_WILLNEED)}) to start fetching that page asynchronously, so
 * that by the time the distance computation begins the data is more likely to be in RAM.
 *
 * <p><b>Prefetch window.</b> A window of {@value #PREFETCH_WINDOW} nodes ahead of the
 * requested node is prefetched on every call.  Prefetching exactly one record provides
 * no overlap; too large a window wastes TLB/cache bandwidth and system-call overhead.
 * The value of 4 matches the typical HNSW node degree (32–64) divided by a conservative
 * overlap factor and is consistent with Lucene's own bulk-score batch size tuning.
 *
 * <p><b>Adaptive skip.</b> {@link OnDiskGraphIndex#prefetchL0Records} delegates to
 * {@link io.github.jbellis.jvector.disk.ReaderSupplier#prefetch}, which in turn calls
 * {@link org.apache.lucene.store.IndexInput#prefetch}.  Lucene's MMap implementation
 * already applies an adaptive counter that skips the {@code madvise} syscall when the
 * page cache appears warm (consecutive hits), so this class does not need its own
 * skip logic.
 *
 * <p>When the underlying directory does not support prefetching (e.g. {@code FSDirectory}
 * on Windows, or non-MMap directories) all calls are silent no-ops.
 */
public final class PrefetchingScoreFunction implements ScoreFunction {

    /**
     * Number of L0 records to hint ahead of the requested node.
     * One record covers: node-id (4 B) + inline features (vector / NVQ codes) + adjacency list.
     * Prefetching 4 ahead provides I/O overlap without over-committing page-cache bandwidth.
     */
    static final int PREFETCH_WINDOW = 4;

    private final ScoreFunction delegate;
    private final OnDiskGraphIndex index;

    /**
     * @param delegate the real score function to delegate to after prefetching
     * @param index    the on-disk graph whose L0 records should be prefetched
     */
    public PrefetchingScoreFunction(ScoreFunction delegate, OnDiskGraphIndex index) {
        this.delegate = delegate;
        this.index = index;
    }

    /**
     * Issues a prefetch for a window of L0 records starting at {@code node}, then
     * delegates to the wrapped {@link ScoreFunction}.
     *
     * <p>The prefetch is fire-and-forget: if the OS or the underlying directory does not
     * support it the call silently returns {@code false} and scoring proceeds normally.
     */
    @Override
    public float similarityTo(int node) {
        // Prefetch node..node+WINDOW-1 before loading node's vector.
        // prefetchL0Records clamps maxNode to idUpperBound-1 internally via the byte-range
        // calculation, so no explicit bounds check is needed here.
        index.prefetchL0Records(node, node + PREFETCH_WINDOW - 1);
        return delegate.similarityTo(node);
    }

    @Override
    public boolean isExact() {
        return delegate.isExact();
    }

    @Override
    public float similarityToNeighbor(int origin, int neighborIndex) {
        return delegate.similarityToNeighbor(origin, neighborIndex);
    }

    @Override
    public void enableSimilarityToNeighbors(int origin) {
        delegate.enableSimilarityToNeighbors(origin);
    }

    @Override
    public boolean supportsSimilarityToNeighbors() {
        return delegate.supportsSimilarityToNeighbors();
    }
}

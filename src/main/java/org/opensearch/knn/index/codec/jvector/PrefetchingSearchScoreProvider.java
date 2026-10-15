/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;

/**
 * A {@link SearchScoreProvider} decorator that wraps the primary {@link ScoreFunction}
 * with a {@link PrefetchingScoreFunction} so that every node lookup during graph search
 * is preceded by a page-cache warm-up hint.
 *
 * <p>The reranker and exactScoreFunction are forwarded unchanged: reranking reads
 * full-precision vectors that are already accessed sequentially (and thus benefit less
 * from random prefetch), while the exact score function is only used when the
 * approximate phase is skipped entirely.
 *
 * <p>Use {@link #wrap(SearchScoreProvider, OnDiskGraphIndex)} instead of the constructor
 * when the supplier returns a {@code null} index (e.g. for in-memory graphs) — the helper
 * returns the original provider unchanged in that case.
 */
public final class PrefetchingSearchScoreProvider implements SearchScoreProvider {

    private final ScoreFunction prefetchingScoreFunction;
    private final SearchScoreProvider delegate;

    private PrefetchingSearchScoreProvider(SearchScoreProvider delegate, OnDiskGraphIndex index) {
        this.delegate = delegate;
        this.prefetchingScoreFunction = new PrefetchingScoreFunction(delegate.scoreFunction(), index);
    }

    /**
     * Returns a {@link SearchScoreProvider} whose {@link #scoreFunction()} issues prefetch
     * hints before every similarity read.  Returns the original {@code ssp} unchanged when:
     * <ul>
     *   <li>{@code index} is {@code null} (in-memory / test graphs), or</li>
     *   <li>the score function supports the fused-neighbor path
     *       ({@link ScoreFunction#supportsSimilarityToNeighbors()} is {@code true}), meaning
     *       neighbor scoring goes through {@link ScoreFunction#similarityToNeighbor} rather
     *       than {@link ScoreFunction#similarityTo} and the prefetch wrapper would be bypassed
     *       entirely anyway.</li>
     * </ul>
     *
     * @param ssp   the original provider built by the field entry
     * @param index the on-disk graph whose L0 records should be prefetched
     */
    public static SearchScoreProvider wrap(SearchScoreProvider ssp, OnDiskGraphIndex index) {
        if (index == null) {
            return ssp;
        }
        // Fused-PQ graphs score neighbors via similarityToNeighbor(), not similarityTo().
        // In that path processNeighbors() never calls similarityTo(), so wrapping it would
        // add overhead without ever firing a prefetch.
        if (ssp.scoreFunction().supportsSimilarityToNeighbors()) {
            return ssp;
        }
        return new PrefetchingSearchScoreProvider(ssp, index);
    }

    @Override
    public ScoreFunction scoreFunction() {
        return prefetchingScoreFunction;
    }

    @Override
    public ScoreFunction.ExactScoreFunction reranker() {
        return delegate.reranker();
    }

    @Override
    public ScoreFunction.ExactScoreFunction exactScoreFunction() {
        return delegate.exactScoreFunction();
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import java.util.Optional;

import io.github.jbellis.jvector.graph.NodeArray;
import io.github.jbellis.jvector.graph.diversity.DiversityProvider;
import io.github.jbellis.jvector.util.BitSet;
import lombok.NonNull;

/**
 * <p>A diversity provider that can be initialized after construction.
 *
 * <p>
 * This is useful during leading segment merge when loading the
 * neighbors score cache from disk. The score cache is loaded as an
 * OnHeapGraphIndex, which requires a diveristy provider during construction.
 * However, creating the diversity provider requires knowledge of the ordinals
 * contained in the graph which is only available after the graph is loaded.
 * Since the diversity provider is not used until the graph is mutated, this
 * class can be used to construct the graph without a diversity provider
 * available.
 */
class DelayedInitDiversityProvider implements DiversityProvider {

    private Optional<DiversityProvider> delegate = Optional.empty();

    /**
     * Creates an uninitialized diversity provider.
     * Call {@link #initialize} before use.
     */
    DelayedInitDiversityProvider() {}

    /** Initialize this DiversityProvider with a delegate */
    void initialize(@NonNull DiversityProvider delegate) {
        if (this.delegate.isPresent()) {
            throw new IllegalStateException("already initialized");
        }
        this.delegate = Optional.of(delegate);
    }

    @Override
    public double retainDiverse(NodeArray neighbors, int maxDegree, int diverseBefore, BitSet selected) {
        if (delegate.isEmpty()) {
            throw new IllegalStateException("DelayedInitDiversityProvider was not initialzied (call initialize() before use)");
        }
        return delegate.get().retainDiverse(neighbors, maxDegree, diverseBefore, selected);
    }
}

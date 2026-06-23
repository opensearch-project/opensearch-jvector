/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import java.io.IOException;
import org.apache.lucene.search.VectorScorer;

/**
 * {@link org.apache.lucene.index.FloatVectorValues} implementation for NVQ-inline segments.
 *
 * <p>Vectors are stored as NVQ-quantized bytes inside the graph nodes; dequantization back to
 * approximate float values is delegated to {@link JVectorIndexQuantization.NVQ}. Because no
 * full-precision vectors are available, {@link #scorer(float[])} is not supported — filtered
 * brute-force searches should not be issued against NVQ-inline segments via this class.
 */
public class JVectorQuantizedNvqVectorValues extends JVectorFloatVectorValues {

    public JVectorQuantizedNvqVectorValues(
        OnDiskGraphIndex onDiskGraphIndex,
        VectorSimilarityFunction similarityFunction,
        GraphNodeIdToDocMap graphNodeIdToDocMap,
        NVQuantization nvqInline
    ) throws IOException {
        super(onDiskGraphIndex, similarityFunction, null, graphNodeIdToDocMap, new JVectorIndexQuantization.NVQ(nvqInline));
    }

    @Override
    public VectorScorer scorer(float[] query) {
        throw new UnsupportedOperationException("vectorScorer() is not supported for NVQ-inline segments");
    }
}

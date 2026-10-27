/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.vector.types.FloatArray;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import java.io.IOException;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.VectorScorer;

public class JVectorFloatVectorValues extends FloatVectorValues {
    public static final int NO_VECTOR = -1;
    static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final OnDiskGraphIndex index;
    private final OnDiskGraphIndex.View view;
    private final VectorSimilarityFunction similarityFunction;
    private final org.apache.lucene.index.VectorSimilarityFunction luceneSimilarityFunction;
    private final int dimension;
    private final int size;
    private final GraphNodeIdToDocMap graphNodeIdToDocMap;
    private final JVectorIndexQuantization quantization;

    public JVectorFloatVectorValues(
        OnDiskGraphIndex onDiskGraphIndex,
        VectorSimilarityFunction similarityFunction,
        org.apache.lucene.index.VectorSimilarityFunction luceneSimilarityFunction,
        GraphNodeIdToDocMap graphNodeIdToDocMap
    ) throws IOException {
        this(onDiskGraphIndex, similarityFunction, luceneSimilarityFunction, graphNodeIdToDocMap, new JVectorIndexQuantization.PQ());
    }

    protected JVectorFloatVectorValues(
        OnDiskGraphIndex onDiskGraphIndex,
        VectorSimilarityFunction similarityFunction,
        org.apache.lucene.index.VectorSimilarityFunction luceneSimilarityFunction,
        GraphNodeIdToDocMap graphNodeIdToDocMap,
        JVectorIndexQuantization quantization
    ) throws IOException {
        this.index = onDiskGraphIndex;
        this.view = index.getView();
        this.dimension = view.dimension();
        this.size = view.size();
        this.similarityFunction = similarityFunction;
        this.luceneSimilarityFunction = luceneSimilarityFunction;
        this.graphNodeIdToDocMap = graphNodeIdToDocMap;
        this.quantization = quantization;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return size;
    }

    public VectorFloat<?> vectorFloatValue(int ord) {
        return quantization.vectorFloatValue(view, ord);
    }

    public DocIndexIterator iterator() {
        return new DocIndexIterator() {
            private int docId = -1;
            private final Bits liveNodes = view.liveNodes();

            @Override
            public long cost() {
                return size();
            }

            @Override
            public int index() {
                return graphNodeIdToDocMap.getJVectorNodeId(docId);
            }

            @Override
            public int docID() {
                return docId;
            }

            @Override
            public int nextDoc() throws IOException {
                // Advance to the next node docId starts from -1 which is why we need to increment docId by 1
                // until maxDoc is reached. If the document has vector field but no value (== null), there will
                // gaps in the document <-> node maps, index() will return NO_VECTOR_OR_DELETED_DOC in such cases.
                while (docId < graphNodeIdToDocMap.getMaxDoc() - 1) {
                    docId++;
                    if (liveNodes.get(docId)) {
                        return docId;
                    }
                }
                docId = NO_MORE_DOCS;

                return docId;
            }

            @Override
            public int advance(int target) throws IOException {
                return slowAdvance(target);
            }
        };
    }

    /**
     * Constructs an iterator that iterates over vectors that have corresponding nodes in the graph (skipping the gaps with non-live / NO_VECTORS nodes).
     * @return an iterator that iterates over vectors that have corresponding nodes in the graph (skipping the gaps with non-live / NO_VECTORS nodes)
     */
    public DocIndexIterator vectorIterator() {
        return new DocIndexIterator() {
            private int docId = -1;
            private final Bits liveNodes = view.liveNodes();

            @Override
            public long cost() {
                return size();
            }

            @Override
            public int index() {
                return graphNodeIdToDocMap.getJVectorNodeId(docId);
            }

            @Override
            public int docID() {
                return docId;
            }

            @Override
            public int nextDoc() throws IOException {
                // Advance to the next node docId starts from -1 which is why we need to increment docId by 1
                // until maxDoc is reached. If the document has vector field but no value (== null), the NO_MORE_DOCS
                // is going to be returned by this method.
                while (docId < graphNodeIdToDocMap.getMaxDoc() - 1) {
                    docId++;
                    if (liveNodes.get(docId) && index() != NO_VECTOR) {
                        return docId;
                    }
                }
                docId = NO_MORE_DOCS;

                return docId;
            }

            @Override
            public int advance(int target) throws IOException {
                return slowAdvance(target);
            }
        };
    }

    @Override
    public float[] vectorValue(int i) throws IOException {
        try {
            final VectorFloat<?> vector = vectorFloatValue(i);

            if (vector instanceof FloatArray fa) {
                return fa.array();
            } else if (vector.get() instanceof float[] arr) {
                return arr;
            }

            float[] v = new float[vector.length()];
            for (int j = 0; j < v.length; j++) {
                v[j] = vector.get(j);
            }

            return v;
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public VectorFloat<?> vectorValueObject(int i) throws IOException {
        return vectorFloatValue(i);
    }

    /**
     * Prefetches the graph node L0 records for the
     * provided ordinals into the OS page cache.
     *
     * @param ordsToPrefetch array of graph node ordinals to prefetch
     * @param numOrds        number of valid entries in {@code ordsToPrefetch}
     */
    @Override
    public void prefetch(int[] ordsToPrefetch, int numOrds) throws IOException {
        if (ordsToPrefetch == null || numOrds <= 0) {
            return;
        }
        final int count = Math.min(numOrds, ordsToPrefetch.length);
        int i = 0;
        while (i < count) {
            final int runStart = i++;
            while (i < count && ordsToPrefetch[i] == ordsToPrefetch[i - 1] + 1) {
                i++;
            }
            // Skip isolated single ordinals — one node is not worth a syscall.
            // Only prefetch runs of 2 or more consecutive ordinals.
            if (i - runStart >= 2) {
                index.prefetchL0Records(ordsToPrefetch[runStart], ordsToPrefetch[i - 1]);
            }
        }
    }

    @Override
    public FloatVectorValues copy() throws IOException {
        return this;
    }

    @Override
    public VectorScorer scorer(float[] query) throws IOException {
        return new JVectorVectorScorer(this, VECTOR_TYPE_SUPPORT.createFloatVector(query), similarityFunction, luceneSimilarityFunction);
    }

}

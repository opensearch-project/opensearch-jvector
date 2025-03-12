/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.VectorScorer;

import java.io.IOException;

public class JVectorFloatVectorValues extends FloatVectorValues {
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final OnDiskGraphIndex onDiskGraphIndex;
    private final OnDiskGraphIndex.View view;
    private final VectorSimilarityFunction similarityFunction;

    public JVectorFloatVectorValues(OnDiskGraphIndex onDiskGraphIndex, VectorSimilarityFunction similarityFunction) throws IOException {
        this.onDiskGraphIndex = onDiskGraphIndex;
        this.view = onDiskGraphIndex.getView();
        this.similarityFunction = similarityFunction;
    }

    @Override
    public int dimension() {
        return onDiskGraphIndex.getDimension();
    }

    @Override
    public int size() {
        return onDiskGraphIndex.size();
    }

    public VectorFloat<?> vectorFloatValue(int ord) {
        if (!onDiskGraphIndex.containsNode(ord)) {
            throw new RuntimeException("ord " + ord + " not found in graph");
        }

        return view.getVector(ord);
    }

    public DocIndexIterator iterator() {
        return new DocIndexIterator() {
            private int docId = -1;
            private final NodesIterator nodesIterator = onDiskGraphIndex.getNodes();

            @Override
            public long cost() {
                return size();
            }

            @Override
            public int index() {
                return docId;
            }

            @Override
            public int docID() {
                return docId;
            }

            @Override
            public int nextDoc() throws IOException {
                if (nodesIterator.hasNext()) {
                    docId = nodesIterator.next();
                } else {
                    docId = NO_MORE_DOCS;
                }

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
            return (float[]) vector.get();
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public FloatVectorValues copy() throws IOException {
        return this;
    }

    @Override
    public VectorScorer scorer(float[] query) throws IOException {
        return new JVectorVectorScorer(this, VECTOR_TYPE_SUPPORT.createFloatVector(query), similarityFunction);
    }

}

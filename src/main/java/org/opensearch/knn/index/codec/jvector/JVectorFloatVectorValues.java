/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class JVectorFloatVectorValues extends FloatVectorValues {
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final OnDiskGraphIndex.View view;
    private final VectorSimilarityFunction similarityFunction;
    private final int dimension;
    private final int size;

    public JVectorFloatVectorValues(OnDiskGraphIndex onDiskGraphIndex, VectorSimilarityFunction similarityFunction) throws IOException {
        this.dimension = onDiskGraphIndex.getDimension();
        this.size = onDiskGraphIndex.getIdUpperBound();
        this.view = onDiskGraphIndex.getView();
        this.similarityFunction = similarityFunction;
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
        VectorFloat<?> value = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
        view.getVectorInto(ord, value, 0);
        return value;
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
                return docId;
            }

            @Override
            public int docID() {
                return docId;
            }

            @Override
            public int nextDoc() throws IOException {
                // Advance to the next node docId starts from -1 which is why we need to increment docId by 1 "size"
                // times
                while (docId < size - 1) {
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

    public BinaryDocValues asBinaryDocValues() {

        final DocIdSetIterator it = iterator();
        final BytesRef bytes = new BytesRef(dimension * Float.BYTES);

        return new BinaryDocValues() {
            @Override
            public BytesRef binaryValue() throws IOException {
                float[] f = vectorValue(docID());
                ByteBuffer bb = ByteBuffer.wrap(bytes.bytes).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < f.length; i++)
                    bb.putFloat(f[i]);

                return bytes;
            }

            @Override
            public boolean advanceExact(int target) throws IOException {
                return it.advance(target) == target;
            }

            @Override
            public int docID() {
                return it.docID();
            }

            @Override
            public int nextDoc() throws IOException {
                return it.nextDoc();
            }

            @Override
            public int advance(int target) throws IOException {
                return it.advance(target);
            }

            @Override
            public long cost() {
                return it.cost();
            }
        };
    }

}

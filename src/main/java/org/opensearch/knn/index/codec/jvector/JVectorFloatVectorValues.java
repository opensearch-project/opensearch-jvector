/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.NVQuantization;
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
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final OnDiskGraphIndex.View view;
    private final VectorSimilarityFunction similarityFunction;
    private final org.apache.lucene.index.VectorSimilarityFunction luceneSimilarityFunction;
    private final int dimension;
    private final int size;
    private final GraphNodeIdToDocMap graphNodeIdToDocMap;
    // Non-null only when the underlying graph stores NVQ vectors inline (no full-resolution vectors available).
    private final NVQuantization nvqInline;
    // Reusable scratch buffer for inline NVQ dequantization (safe: access is serialised by the caller).
    private final NVQuantization.QuantizedVector nvqScratch;

    /** Full-precision (non-NVQ) constructor — Lucene similarity available for vectorScorer(). */
    public JVectorFloatVectorValues(
        OnDiskGraphIndex onDiskGraphIndex,
        VectorSimilarityFunction similarityFunction,
        org.apache.lucene.index.VectorSimilarityFunction luceneSimilarityFunction,
        GraphNodeIdToDocMap graphNodeIdToDocMap
    ) throws IOException {
        this(onDiskGraphIndex, similarityFunction, luceneSimilarityFunction, graphNodeIdToDocMap, null);
    }

    /** NVQ-inline constructor — luceneSimilarityFunction not available; vectorScorer() unsupported. */
    public JVectorFloatVectorValues(
        OnDiskGraphIndex onDiskGraphIndex,
        VectorSimilarityFunction similarityFunction,
        GraphNodeIdToDocMap graphNodeIdToDocMap,
        NVQuantization nvqInline
    ) throws IOException {
        this(onDiskGraphIndex, similarityFunction, null, graphNodeIdToDocMap, nvqInline);
    }

    private JVectorFloatVectorValues(
        OnDiskGraphIndex onDiskGraphIndex,
        VectorSimilarityFunction similarityFunction,
        org.apache.lucene.index.VectorSimilarityFunction luceneSimilarityFunction,
        GraphNodeIdToDocMap graphNodeIdToDocMap,
        NVQuantization nvqInline
    ) throws IOException {
        this.view = onDiskGraphIndex.getView();
        this.dimension = view.dimension();
        this.size = view.size();
        this.similarityFunction = similarityFunction;
        this.luceneSimilarityFunction = luceneSimilarityFunction;
        this.graphNodeIdToDocMap = graphNodeIdToDocMap;
        this.nvqInline = nvqInline;
        this.nvqScratch = nvqInline != null
            ? NVQuantization.QuantizedVector.createEmpty(nvqInline.subvectorSizesAndOffsets, nvqInline.bitsPerDimension)
            : null;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return size;
    }

    /**
     * Returns the float vector for the given graph-node ordinal.
     *
     * For normal (non-NVQ-inline) graphs the full-resolution vector is read directly.
     * For NVQ-inline graphs the quantized bytes stored in the graph node are
     * dequantized back to approximate float values so that callers (e.g. the merge
     * path) can treat all segments uniformly.
     */
    public VectorFloat<?> vectorFloatValue(int ord) {
        if (nvqInline != null) {
            return dequantizeNVQInline(ord);
        }
        return view.getVector(ord);
    }

    // -------------------------------------------------------------------------
    // NVQ inline dequantization helpers
    // -------------------------------------------------------------------------

    private VectorFloat<?> dequantizeNVQInline(int ord) {
        try {
            var reader = view.featureReaderForNode(ord, FeatureId.NVQ_VECTORS);
            NVQuantization.QuantizedVector.loadInto(reader, nvqScratch);
            return nvqDequantize(nvqScratch, nvqInline);
        } catch (IOException e) {
            throw new RuntimeException("Failed to dequantize NVQ inline vector for ordinal " + ord, e);
        }
    }

    /**
     * Reconstructs an approximate float vector from a {@link NVQuantization.QuantizedVector}.
     *
     * The inverse of the NVQ encoding is:
     * <pre>
     *   scaledValue   = b * logisticScale + logisticBias      (map byte → sigmoid domain)
     *   reconstructed = logit(scaledValue) / scaledGrowthRate + scaledMidpoint
     * </pre>
     * followed by adding back the {@code globalMean} that was subtracted during encoding.
     */
    private static VectorFloat<?> nvqDequantize(NVQuantization.QuantizedVector qv, NVQuantization nvq) {
        VectorFloat<?> result = VECTOR_TYPE_SUPPORT.createFloatVector(nvq.originalDimension);
        int offset = 0;
        for (NVQuantization.QuantizedSubVector sv : qv.subVectors) {
            float delta = sv.maxValue - sv.minValue;
            float scaledGrowthRate = sv.growthRate / delta;
            float scaledMidpoint = sv.midpoint * delta;
            float logisticBias = logisticNQT(sv.minValue, scaledGrowthRate, scaledMidpoint);
            float logisticScale = (logisticNQT(sv.maxValue, scaledGrowthRate, scaledMidpoint) - logisticBias) / 255.0f;
            float inverseScaledGrowthRate = 1.0f / scaledGrowthRate;

            for (int d = 0; d < sv.originalDimensions; d++) {
                int b = Byte.toUnsignedInt(sv.bytes.get(d));
                float scaledValue = Math.fma(b, logisticScale, logisticBias);
                result.set(offset + d, logitNQT(scaledValue, inverseScaledGrowthRate, scaledMidpoint));
            }
            offset += sv.originalDimensions;
        }
        // Add back global mean (subtracted during encoding)
        for (int i = 0; i < nvq.originalDimension; i++) {
            result.set(i, result.get(i) + nvq.globalMean.get(i));
        }
        return result;
    }

    /** Fast logistic function used by NVQ (mirrors DefaultVectorUtilSupport.logisticFunctionNQT). */
    private static float logisticNQT(float value, float alpha, float x0) {
        float temp = Math.fma(value, alpha, -alpha * x0);
        int p = Math.round(temp + 0.5f);
        int m = Float.floatToIntBits(Math.fma(temp - p, 0.5f, 1));
        temp = Float.intBitsToFloat(m + (p << 23));
        return temp / (temp + 1);
    }

    /** Fast logit (inverse logistic) used by NVQ (mirrors DefaultVectorUtilSupport.logitNQT). */
    private static float logitNQT(float scaledValue, float inverseAlpha, float x0) {
        float z = scaledValue / (1 - scaledValue);
        int temp = Float.floatToIntBits(z);
        int e = temp & 0x7f800000;
        float p = (float) ((e >> 23) - 128);
        float m = Float.intBitsToFloat((temp & 0x007fffff) + 0x3f800000);
        return (m + p) * inverseAlpha + x0;
    }

    // -------------------------------------------------------------------------

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
            return (float[]) vector.get();
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public VectorFloat<?> vectorValueObject(int i) throws IOException {
        return vectorFloatValue(i);
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

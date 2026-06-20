/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.io.IOException;
import org.apache.lucene.search.VectorScorer;

/**
 * {@link org.apache.lucene.index.FloatVectorValues} implementation for NVQ-inline segments.
 *
 * <p>Vectors are stored as NVQ-quantized bytes inside the graph nodes and are dequantized back to
 * approximate float values on demand.  Because no full-precision vectors are available,
 * {@link #scorer(float[])} is not supported — filtered brute-force searches should not be issued
 * against NVQ-inline segments via this class.
 */
public class JVectorQuantizedNvqVectorValues extends JVectorFloatVectorValues {

    private final NVQuantization nvqInline;
    // Reusable scratch buffer; access must be serialised by the caller.
    private final NVQuantization.QuantizedVector nvqTemp;

    public JVectorQuantizedNvqVectorValues(
        OnDiskGraphIndex onDiskGraphIndex,
        VectorSimilarityFunction similarityFunction,
        GraphNodeIdToDocMap graphNodeIdToDocMap,
        NVQuantization nvqInline
    ) throws IOException {
        super(onDiskGraphIndex, similarityFunction, null, graphNodeIdToDocMap);
        this.nvqInline = nvqInline;
        this.nvqTemp = NVQuantization.QuantizedVector.createEmpty(nvqInline.subvectorSizesAndOffsets, nvqInline.bitsPerDimension);
    }

    /**
     * Dequantizes the NVQ-encoded bytes for {@code ord} back to approximate float values.
     */
    @Override
    public VectorFloat<?> vectorFloatValue(int ord) {
        try {
            var reader = getView().featureReaderForNode(ord, FeatureId.NVQ_VECTORS);
            NVQuantization.QuantizedVector.loadInto(reader, nvqTemp);
            return nvqDequantize(nvqTemp, nvqInline);
        } catch (IOException e) {
            throw new RuntimeException("Failed to dequantize NVQ inline vector for ordinal " + ord, e);
        }
    }

    @Override
    public VectorScorer scorer(float[] query) {
        throw new UnsupportedOperationException("vectorScorer() is not supported for NVQ-inline segments");
    }

    // -------------------------------------------------------------------------
    // NVQ dequantization helpers
    // -------------------------------------------------------------------------

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
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import java.io.IOException;
import java.util.function.Function;
import org.opensearch.knn.common.KNNConstants;

/**
 * Encapsulates the quantization strategy used when writing a jVector segment.
 * <p>
 * Concrete subclasses are {@link NVQ} for Non-uniform Vector Quantization (stored inline
 * in the graph) and {@link PQ} for Product Quantization (stored as a separate blob).
 * PQ and NVQ quantize different portions of the graph. As a result,
 * the following combinations for quantization are possible today in the disk-ann graph:
 * 1. PQ + full-precision vectors
 * 2. PQ + NVQ
 */
public sealed abstract class JVectorIndexQuantization {

    static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    /** String identifier used in REST params and on-disk serialization. */
    public abstract String getType();

    /** Returns the number of subspaces/subvectors to use for a vector of the given dimension. */
    public abstract int numSubspaces(int dimension);

    /** Reads back (and, for lossy schemes, dequantizes) the float vector stored for {@code ord} in {@code view}. */
    public abstract VectorFloat<?> vectorFloatValue(OnDiskGraphIndex.View view, int ord);

    // -----------------------------------------------------------------------
    // NVQ implementation
    // -----------------------------------------------------------------------

    public static final class NVQ extends JVectorIndexQuantization {
        private final int numSubvectors;
        // Bound only when this instance is used to read back (dequantize) stored vectors; null at write/config time.
        private final NVQuantization trainedQuantizer;
        private final NVQuantization.QuantizedVector scratch;

        public NVQ(int numSubvectors) {
            this.numSubvectors = numSubvectors;
            this.trainedQuantizer = null;
            this.scratch = null;
        }

        /** Binds a trained {@link NVQuantization} so this instance can dequantize stored vectors. */
        public NVQ(NVQuantization trainedQuantizer) {
            this.numSubvectors = trainedQuantizer.subvectorSizesAndOffsets.length;
            this.trainedQuantizer = trainedQuantizer;
            this.scratch = NVQuantization.QuantizedVector.createEmpty(
                trainedQuantizer.subvectorSizesAndOffsets,
                trainedQuantizer.bitsPerDimension
            );
        }

        public int getNumSubvectors() {
            return numSubvectors;
        }

        @Override
        public String getType() {
            return KNNConstants.QUANTIZATION_TYPE_NVQ;
        }

        @Override
        public int numSubspaces(int dimension) {
            return numSubvectors;
        }

        /**
         * Dequantizes the NVQ-encoded bytes for {@code ord} back to approximate float values.
         */
        @Override
        public VectorFloat<?> vectorFloatValue(OnDiskGraphIndex.View view, int ord) {
            if (trainedQuantizer == null) {
                throw new IllegalStateException("NVQ quantization is not bound to a trained quantizer; cannot dequantize");
            }
            try {
                var reader = view.featureReaderForNode(ord, FeatureId.NVQ_VECTORS);
                NVQuantization.QuantizedVector.loadInto(reader, scratch);
                return nvqDequantize(scratch, trainedQuantizer);
            } catch (IOException e) {
                throw new RuntimeException("Failed to dequantize NVQ inline vector for ordinal " + ord, e);
            }
        }

        /**
         * Returns the recommended number of NVQ subvectors for a given vector dimension.
         *
         * <p>Unlike PQ (which stores only 1 byte per subvector as a centroid index), each NVQ
         * subvector carries 28 bytes of fixed overhead: four floats (growthRate, midpoint, minValue,
         * maxValue) and three ints for the sigmoid parameterization. To keep the NVQ inline graph
         * smaller than full-precision storage ({@code dim × 4} bytes), M must satisfy:
         *
         * <pre>  4 + dim + 28 × M  &lt;  4 × dim   →   M  &lt;  3 × dim / 28</pre>
         *
         * The default value of 2 subvectors is set based on a recommendation by the jvector
         * library presently.
         *
         * @return the number of NVQ subvectors (at least 1)
         */
        public static int defaultNumSubvectors() {
            return 2;
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

    // -----------------------------------------------------------------------
    // PQ implementation
    // -----------------------------------------------------------------------

    public static final class PQ extends JVectorIndexQuantization {
        private final Function<Integer, Integer> numSubspacesSupplier;

        public PQ(Function<Integer, Integer> numSubspacesSupplier) {
            this.numSubspacesSupplier = numSubspacesSupplier;
        }

        /** PQ with a fixed (dimension-independent) subspace count. */
        public PQ(int fixedNumSubspaces) {
            this(ignored -> fixedNumSubspaces);
        }

        /** PQ using the default dimension-adaptive subspace count. */
        public PQ() {
            this(PQ::defaultNumSubspaces);
        }

        public Function<Integer, Integer> getNumSubspacesSupplier() {
            return numSubspacesSupplier;
        }

        @Override
        public String getType() {
            return KNNConstants.QUANTIZATION_TYPE_PQ;
        }

        @Override
        public int numSubspaces(int dimension) {
            return numSubspacesSupplier.apply(dimension);
        }

        /** PQ does not quantize the inline graph vectors; they are read back at full precision. */
        @Override
        public VectorFloat<?> vectorFloatValue(OnDiskGraphIndex.View view, int ord) {
            return view.getVector(ord);
        }

        /**
         * Returns the default number of PQ subspaces for a given vector dimension.
         *
         * <p>The idea is that higher dimensions compress well, but not so well that we should use
         * fewer bits than a lower-dimension vector, which is what you could get with cutoff points
         * to switch between (e.g.) D*0.5 and D*0.25. Thus, the following ensures that bytes per
         * vector is strictly increasing with D.
         *
         * @param originalDimension original vector dimension
         * @return default number of subspaces per vector
         */
        public static int defaultNumSubspaces(int originalDimension) {
            int compressedBytes;
            if (originalDimension <= 32) {
                compressedBytes = originalDimension;
            } else if (originalDimension <= 64) {
                compressedBytes = 32;
            } else if (originalDimension <= 200) {
                compressedBytes = (int) (originalDimension * 0.5);
            } else if (originalDimension <= 400) {
                compressedBytes = 100;
            } else if (originalDimension <= 768) {
                compressedBytes = (int) (originalDimension * 0.25);
            } else if (originalDimension <= 1536) {
                compressedBytes = 192;
            } else {
                compressedBytes = (int) (originalDimension * 0.125);
            }
            return compressedBytes;
        }
    }
}

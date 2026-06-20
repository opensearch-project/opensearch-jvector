/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

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
public abstract class JVectorIndexQuantization {

    /** String identifier used in REST params and on-disk serialization. */
    public abstract String getType();

    /** Returns the number of subspaces/subvectors to use for a vector of the given dimension. */
    public abstract int numSubspaces(int dimension);

    // -----------------------------------------------------------------------
    // NVQ implementation
    // -----------------------------------------------------------------------

    public static final class NVQ extends JVectorIndexQuantization {
        private final int numSubvectors;

        public NVQ(int numSubvectors) {
            this.numSubvectors = numSubvectors;
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

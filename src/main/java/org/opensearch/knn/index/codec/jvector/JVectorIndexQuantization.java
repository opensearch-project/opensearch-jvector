/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.CompressedVectors;
import io.github.jbellis.jvector.quantization.NVQVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import java.io.IOException;
import java.time.Clock;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Function;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.plugin.stats.KNNCounter;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;

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
public sealed interface JVectorIndexQuantization {

    Logger log = LogManager.getLogger(JVectorIndexQuantization.class);
    static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    // On-disk quantization type bytes written into VectorIndexFieldMetadata
    byte QUANTIZATION_TYPE_NONE = 0;
    byte QUANTIZATION_TYPE_PQ = 1;
    byte QUANTIZATION_TYPE_NVQ_INLINE = 2;

    /** Holds the quantization objects loaded from disk for a single field. */
    record LoadedState(NVQuantization nvqInlineQuantization, PQVectors pqVectors, ReaderSupplier compressedVectorsReaderSupplier) {
    }

    /** Result of quantizing a set of vectors ahead of graph construction. */
    record QuantizationResult(CompressedVectors compressedVectors, PQVectors auxiliaryPqVectors, BuildScoreProvider buildScoreProvider) {
    }

    /**
     * Loads the quantization state for a field from disk, dispatching on the on-disk {@code qType} byte.
     * For NVQ, the {@link NVQuantization} is extracted from the graph and an optional auxiliary PQ blob is loaded.
     * For PQ-only, the PQ blob is loaded from the field data file.
     */
    static LoadedState loadQuantizationState(
        byte qType,
        OnDiskGraphIndex index,
        Directory directory,
        String fieldDataFileName,
        long compressedVectorsOffset,
        long compressedVectorsLength,
        long vectorIndexOffset
    ) throws IOException {
        return switch (qType) {
            case QUANTIZATION_TYPE_NVQ_INLINE -> loadNVQState(
                index,
                directory,
                fieldDataFileName,
                compressedVectorsOffset,
                compressedVectorsLength
            );
            default -> compressedVectorsLength > 0
                ? loadPQState(directory, fieldDataFileName, compressedVectorsOffset, compressedVectorsLength, vectorIndexOffset)
                : new LoadedState(null, null, null);
        };
    }

    /**
     * Compute NVQ vectors from raw float vectors.
     * Called by {@link JVectorIndexQuantization.NVQ#quantize} and directly from the merge path.
     */
    static NVQVectors computeNvqVectors(int numSubvectors, RandomAccessVectorValues ravv, ForkJoinPool computePool) throws IOException {
        log.info("Computing NVQ parameters for {} vectors", ravv.size());
        final long start = Clock.systemDefaultZone().millis();
        NVQuantization nvq = NVQuantization.compute(ravv, numSubvectors);
        KNNCounter.KNN_QUANTIZATION_TRAINING_TIME.add(Clock.systemDefaultZone().millis() - start);
        log.info("Encoding NVQ vectors for {} vectors", ravv.size());
        NVQVectors nvqVectors = nvq.encodeAll(ravv, computePool);
        log.info(
            "Encoded NVQ vectors, original size: {} bytes, compressed size: {} bytes",
            nvqVectors.getOriginalSize(),
            nvqVectors.getCompressedSize()
        );
        return nvqVectors;
    }

    /**
     * Compute PQ vectors from raw float vectors with the given subspace count.
     * Called by {@link #quantize} implementations and directly from the merge path.
     */
    static PQVectors computePqVectors(
        RandomAccessVectorValues ravv,
        VectorSimilarityFunction vsf,
        int numSubspaces,
        ForkJoinPool computePool
    ) throws IOException {
        log.info("Computing PQ codebooks for {} vectors", ravv.size());
        final long start = Clock.systemDefaultZone().millis();
        final int clusters = Math.min(256, ravv.size());
        ProductQuantization pq = ProductQuantization.compute(
            ravv,
            numSubspaces,
            clusters,
            vsf == VectorSimilarityFunction.EUCLIDEAN,
            UNWEIGHTED,
            computePool,
            ForkJoinPool.commonPool()
        );
        KNNCounter.KNN_QUANTIZATION_TRAINING_TIME.add(Clock.systemDefaultZone().millis() - start);
        PQVectors pqVectors = PQVectors.encodeAndBuild(pq, ravv.size(), ravv, computePool);
        log.info(
            "Encoded PQ vectors, original size: {} bytes, compressed size: {} bytes",
            pqVectors.getOriginalSize(),
            pqVectors.getCompressedSize()
        );
        return pqVectors;
    }

    private static LoadedState loadNVQState(
        OnDiskGraphIndex index,
        Directory directory,
        String fieldDataFileName,
        long compressedVectorsOffset,
        long compressedVectorsLength
    ) throws IOException {
        NVQuantization nvq = nvqFromGraph(index);
        if (compressedVectorsLength == 0) {
            return new LoadedState(nvq, null, null);
        }
        assert compressedVectorsOffset > 0;
        ReaderSupplier supplier = new JVectorRandomAccessReader.Supplier(
            directory.openInput(fieldDataFileName, IOContext.READONCE),
            compressedVectorsOffset,
            compressedVectorsLength
        );
        PQVectors pqVectors;
        try (var reader = supplier.get()) {
            pqVectors = PQVectors.load(reader);
        }
        return new LoadedState(nvq, pqVectors, supplier);
    }

    private static LoadedState loadPQState(
        Directory directory,
        String fieldDataFileName,
        long compressedVectorsOffset,
        long compressedVectorsLength,
        long vectorIndexOffset
    ) throws IOException {
        if (compressedVectorsOffset < vectorIndexOffset) {
            throw new IllegalArgumentException("compressedVectorsOffset must be greater than vectorIndexOffset");
        }
        ReaderSupplier supplier = new JVectorRandomAccessReader.Supplier(
            directory.openInput(fieldDataFileName, IOContext.READONCE),
            compressedVectorsOffset,
            compressedVectorsLength
        );
        PQVectors pqVectors;
        try (var reader = supplier.get()) {
            pqVectors = PQVectors.load(reader);
        }
        return new LoadedState(null, pqVectors, supplier);
    }

    // TODO: replace reflection with nvqFeature.getNVQuantization() once jvector exposes it publicly in the released jar
    private static NVQuantization nvqFromGraph(OnDiskGraphIndex index) throws IOException {
        io.github.jbellis.jvector.graph.disk.feature.NVQ nvqFeature = (io.github.jbellis.jvector.graph.disk.feature.NVQ) index.getFeatures()
            .get(FeatureId.NVQ_VECTORS);
        if (nvqFeature == null) {
            return null;
        }
        try {
            java.lang.reflect.Field nvqField = io.github.jbellis.jvector.graph.disk.feature.NVQ.class.getDeclaredField("nvq");
            nvqField.setAccessible(true);
            return (NVQuantization) nvqField.get(nvqFeature);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new IOException("Unable to extract NVQuantization from NVQ feature via reflection", e);
        }
    }

    /** String identifier used in REST params and on-disk serialization. */
    public abstract String getType();

    /** Returns the byte value written to the segment metadata to identify this quantization type. */
    public abstract byte quantizationType();

    /** Returns the number of subspaces/subvectors to use for a vector of the given dimension. */
    public abstract int numSubspaces(int dimension);

    /** Reads back (and, for lossy schemes, dequantizes) the float vector stored for {@code ord} in {@code view}. */
    public abstract VectorFloat<?> vectorFloatValue(OnDiskGraphIndex.View view, int ord);

    /**
     * Compute compressed vectors and a build score provider from raw float vectors.
     * Dispatches to the NVQ or PQ implementation based on the concrete type.
     */
    public abstract QuantizationResult quantize(RandomAccessVectorValues ravv, VectorSimilarityFunction vsf, ForkJoinPool computePool)
        throws IOException;

    // -----------------------------------------------------------------------
    // NVQ implementation
    // -----------------------------------------------------------------------

    public static final class NVQ implements JVectorIndexQuantization {
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
        public byte quantizationType() {
            return QUANTIZATION_TYPE_NVQ_INLINE;
        }

        @Override
        public int numSubspaces(int dimension) {
            return numSubvectors;
        }

        @Override
        public QuantizationResult quantize(RandomAccessVectorValues ravv, VectorSimilarityFunction vsf, ForkJoinPool computePool)
            throws IOException {
            NVQVectors nvqVectors = computeNvqVectors(numSubvectors, ravv, computePool);
            PQVectors auxPqVectors = computePqVectors(ravv, vsf, PQ.defaultNumSubspaces(ravv.dimension()), computePool);
            return new QuantizationResult(nvqVectors, auxPqVectors, BuildScoreProvider.pqBuildScoreProvider(vsf, auxPqVectors));
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

    public static final class PQ implements JVectorIndexQuantization {
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
        public byte quantizationType() {
            return QUANTIZATION_TYPE_PQ;
        }

        @Override
        public int numSubspaces(int dimension) {
            return numSubspacesSupplier.apply(dimension);
        }

        @Override
        public QuantizationResult quantize(RandomAccessVectorValues ravv, VectorSimilarityFunction vsf, ForkJoinPool computePool)
            throws IOException {
            PQVectors pqVectors = computePqVectors(ravv, vsf, numSubspaces(ravv.dimension()), computePool);
            return new QuantizationResult(pqVectors, null, BuildScoreProvider.pqBuildScoreProvider(vsf, pqVectors));
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

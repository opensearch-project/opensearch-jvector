/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import java.io.Closeable;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.*;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.*;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.io.Closeable;
import java.io.IOException;
import java.lang.reflect.Field;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;


@Log4j2
public class JVectorReader extends KnnVectorsReader {
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final FieldInfos fieldInfos;
    private final String baseDataFileName;
    // Maps field name to field entries
    private final Map<String, FieldEntry> fieldEntryMap = new HashMap<>(1);
    private final Directory directory;
    private final SegmentReadState state;

    public JVectorReader(SegmentReadState state) throws IOException {
        this.state = state;
        this.fieldInfos = state.fieldInfos;
        this.baseDataFileName = state.segmentInfo.name + "_" + state.segmentSuffix;
        final String metaFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            JVectorFormat.META_EXTENSION
        );
        this.directory = state.directory;
        boolean success = false;
        try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
            final int version = CodecUtil.checkIndexHeader(
                meta,
                JVectorFormat.META_CODEC_NAME,
                JVectorFormat.VERSION_START,
                JVectorFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            readFields(meta, version);
            CodecUtil.checkFooter(meta);

            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    @Override
    public void checkIntegrity() throws IOException {
        for (FieldEntry fieldEntry : fieldEntryMap.values()) {
            // Verify the vector index file
            try (var indexInput = state.directory.openInput(fieldEntry.vectorIndexFieldDataFileName, IOContext.READONCE)) {
                CodecUtil.checksumEntireFile(indexInput);
            }

            // Verify the neighbors score cache file
            try (var indexInput = state.directory.openInput(fieldEntry.neighborsScoreCacheIndexFieldFileName, IOContext.READONCE)) {
                CodecUtil.checksumEntireFile(indexInput);
            }
        }
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        final FieldEntry fieldEntry = fieldEntryMap.get(field);
        return new JVectorFloatVectorValues(
            fieldEntry.index,
            fieldEntry.similarityFunction,
            fieldEntry.graphNodeIdToDocMap,
            fieldEntry.nvqInlineQuantization
        );
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        /**
         * Byte vector values are not supported in jVector library. Instead use PQ.
         */
        return null;
    }

    public Optional<ProductQuantization> getProductQuantizationForField(String field) throws IOException {
        final FieldEntry fieldEntry = fieldEntryMap.get(field);
        if (fieldEntry.pqVectors == null) {
            return Optional.empty();
        }

        return Optional.of(fieldEntry.pqVectors.getCompressor());
    }

    public Optional<NVQuantization> getNVQuantizationForField(String field) throws IOException {
        final FieldEntry fieldEntry = fieldEntryMap.get(field);
        return Optional.ofNullable(fieldEntry.nvqInlineQuantization);
    }

    public RandomAccessReader getNeighborsScoreCacheForField(String field) throws IOException {
        final FieldEntry fieldEntry = fieldEntryMap.get(field);
        return fieldEntry.neighborsScoreCacheIndexReaderSupplier.get();
    }

    public OnDiskGraphIndex getOnDiskGraphIndex(String field) throws IOException {
        return fieldEntryMap.get(field).index;
    }

    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        final OnDiskGraphIndex index = fieldEntryMap.get(field).index;
        final JVectorKnnCollector jvectorKnnCollector;
        if (knnCollector instanceof JVectorKnnCollector) {
            jvectorKnnCollector = (JVectorKnnCollector) knnCollector;
        } else {
            log.warn("KnnCollector must be of type JVectorKnnCollector, for now we will re-wrap it but this is not ideal");
            jvectorKnnCollector = new JVectorKnnCollector(
                knnCollector,
                KNNConstants.DEFAULT_QUERY_SIMILARITY_THRESHOLD.floatValue(),
                KNNConstants.DEFAULT_QUERY_RERANK_FLOOR.floatValue(),
                KNNConstants.DEFAULT_OVER_QUERY_FACTOR,
                KNNConstants.DEFAULT_QUERY_USE_PRUNING
            );

        }

        // search for a random vector using a GraphSearcher and SearchScoreProvider
        VectorFloat<?> q = VECTOR_TYPE_SUPPORT.createFloatVector(target);
        final SearchScoreProvider ssp;

        // Get the Lucene similarity function to check if we need to transform scores
        final FieldEntry fieldEntry = fieldEntryMap.get(field);
        final org.apache.lucene.index.VectorSimilarityFunction luceneSimilarityFunction = fieldEntry.fieldInfo
            .getVectorSimilarityFunction();
        final VectorSimilarityFunction vectorSimilarityFunction = fieldEntry.similarityFunction;

        try (var view = index.getView()) {
            final long graphSearchStart = System.currentTimeMillis();
            final FieldEntry fe = fieldEntryMap.get(field);
            if (fe.pqVectors != null) {
                // PQ blob as approximate first pass; reranker reads inline vectors (NVQ or full-precision)
                ScoreFunction.ApproximateScoreFunction asf = fe.pqVectors.precomputedScoreFunctionFor(q, fe.similarityFunction);
                ScoreFunction.ExactScoreFunction reranker = view.rerankerFor(q, fe.similarityFunction);
                ssp = new DefaultSearchScoreProvider(asf, reranker);
            } else if (fe.nvqInlineQuantization != null) { // NVQ inline without PQ blob
                ssp = new DefaultSearchScoreProvider(view.rerankerFor(q, fe.similarityFunction));
            } else {
                ssp = DefaultSearchScoreProvider.exact(q, fe.similarityFunction, view);
            }
            final GraphNodeIdToDocMap jvectorLuceneDocMap = fe.graphNodeIdToDocMap;
            // Convert the acceptDocs bitmap from Lucene to jVector ordinal bitmap filter
            // Logic works as follows: if acceptDocs is null, we accept all ordinals. Otherwise, we check if the jVector ordinal has a
            // corresponding Lucene doc ID accepted by acceptDocs filter.
            io.github.jbellis.jvector.util.Bits compatibleBits;
            if (acceptDocs == null) compatibleBits = ord -> true;
            else {
                Bits b = acceptDocs.bits();
                compatibleBits = ord -> b == null
                    || (jvectorLuceneDocMap.getLuceneDocId(ord) != -1 && b.get(jvectorLuceneDocMap.getLuceneDocId(ord)));
            }

            try (var graphSearcher = new GraphSearcher(index)) {
                final var searchResults = graphSearcher.search(
                    ssp,
                    jvectorKnnCollector.k(),
                    jvectorKnnCollector.k() * jvectorKnnCollector.getOverQueryFactor(),
                    jvectorKnnCollector.getThreshold(),
                    jvectorKnnCollector.getRerankFloor(),
                    compatibleBits
                );

                for (SearchResult.NodeScore ns : searchResults.getNodes()) {
                    jvectorKnnCollector.collect(jvectorLuceneDocMap.getLuceneDocId(ns.node), ns.score);
                }
                final long graphSearchEnd = System.currentTimeMillis();
                final long searchTime = graphSearchEnd - graphSearchStart;
                log.debug("Search (including acquiring view) took {} ms", searchTime);

                // Collect the below metrics about the search and somehow wire this back to {@link @KNNStats}
                final int visitedNodesCount = searchResults.getVisitedCount();
                final int rerankedCount = searchResults.getRerankedCount();

                final int expandedCount = searchResults.getExpandedCount();
                final int expandedBaseLayerCount = searchResults.getExpandedCountBaseLayer();

                KNNCounter.KNN_QUERY_VISITED_NODES.add(visitedNodesCount);
                KNNCounter.KNN_QUERY_RERANKED_COUNT.add(rerankedCount);
                KNNCounter.KNN_QUERY_EXPANDED_NODES.add(expandedCount);
                KNNCounter.KNN_QUERY_EXPANDED_BASE_LAYER_NODES.add(expandedBaseLayerCount);
                KNNCounter.KNN_QUERY_GRAPH_SEARCH_TIME.add(searchTime);
                log.debug(
                    "rerankedCount: {}, visitedNodesCount: {}, expandedCount: {}, expandedBaseLayerCount: {}",
                    rerankedCount,
                    visitedNodesCount,
                    expandedCount,
                    expandedBaseLayerCount
                );

                // Apache Lucene tracks visited counter so to validate scored docs/ total hits (
                // see AbstractKnnVectorQuery please). The counter has to be updated manually.
                final int visitedCount = visitedNodesCount + expandedCount;
                if (visitedCount > 0) {
                    jvectorKnnCollector.incVisitedCount(visitedCount);
                }
            }
        }
    }

    /**
     * Wraps an ExactScoreFunction to handle score transformation between jVector and Lucene similarity functions for innerproduct.
     *
     * @param delegate the base ExactScoreFunction to wrap
     * @param luceneSimilarityFunction the Lucene similarity function
     * @param jvectorSimilarityFunction the jVector similarity function
     * @return a wrapped ExactScoreFunction that applies score transformation if needed
     */
    private static ScoreFunction.ExactScoreFunction wrapExactScoreFunction(
        ScoreFunction.ExactScoreFunction delegate,
        org.apache.lucene.index.VectorSimilarityFunction luceneSimilarityFunction,
        VectorSimilarityFunction jvectorSimilarityFunction
    ) {
        // Lucene's MAXIMUM_INNER_PRODUCT formula is: 1 + dotProduct
        // jVector's DOT_PRODUCT returns: (1 + dotProduct) / 2
        // To convert: score * 2 = (1 + dotProduct) / 2 * 2 = 1 + dotProduct
        if (luceneSimilarityFunction == org.apache.lucene.index.VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
            && jvectorSimilarityFunction == VectorSimilarityFunction.DOT_PRODUCT) {
            return new ScoreFunction.ExactScoreFunction() {
                @Override
                public float similarityTo(int node2) {
                    return delegate.similarityTo(node2) * 2.0f;
                }
            };
        } else {
            return delegate;
        }
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        // TODO: implement this
        throw new UnsupportedOperationException("Byte vector search is not supported yet with jVector");
    }

    @Override
    public void close() throws IOException {
        for (FieldEntry fieldEntry : fieldEntryMap.values()) {
            IOUtils.close(fieldEntry);
        }
        fieldEntryMap.clear();
    }

    private void readFields(ChecksumIndexInput meta, int version) throws IOException {
        for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
            final FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldNumber); // read field number
            JVectorWriter.VectorIndexFieldMetadata vectorIndexFieldMetadata = new JVectorWriter.VectorIndexFieldMetadata(meta, version);
            assert fieldInfo.number == vectorIndexFieldMetadata.getFieldNumber();
            fieldEntryMap.put(fieldInfo.name, new FieldEntry(fieldInfo, vectorIndexFieldMetadata));
        }
    }

    class FieldEntry implements Closeable {
        private final FieldInfo fieldInfo;
        private final VectorEncoding vectorEncoding;
        private final VectorSimilarityFunction similarityFunction;
        private final int dimension;
        private final long vectorIndexOffset;
        private final long vectorIndexLength;
        private final long compressedVectorsLength;
        private final long compressedVectorsOffset;
        private final String vectorIndexFieldDataFileName;
        private final String neighborsScoreCacheIndexFieldFileName;
        private final GraphNodeIdToDocMap graphNodeIdToDocMap;
        private final ReaderSupplier indexReaderSupplier;
        private final ReaderSupplier compressedVectorsReaderSupplier;
        private final ReaderSupplier neighborsScoreCacheIndexReaderSupplier;
        private final OnDiskGraphIndex index;
        private final PQVectors pqVectors; // non-null when a PQ blob is present (PQ-only or NVQ+PQ)
        // NVQuantization extracted from the graph when NVQ is stored inline; null otherwise
        final NVQuantization nvqInlineQuantization;

        public FieldEntry(FieldInfo fieldInfo, JVectorWriter.VectorIndexFieldMetadata vectorIndexFieldMetadata) throws IOException {
            this.fieldInfo = fieldInfo;
            this.similarityFunction = VectorSimilarityMapper.ordToDistFunc(
                vectorIndexFieldMetadata.getVectorSimilarityFunction().ordinal()
            );
            this.vectorEncoding = vectorIndexFieldMetadata.getVectorEncoding();
            this.vectorIndexOffset = vectorIndexFieldMetadata.getVectorIndexOffset();
            this.vectorIndexLength = vectorIndexFieldMetadata.getVectorIndexLength();
            this.compressedVectorsLength = vectorIndexFieldMetadata.getCompressedVectorsLength();
            this.compressedVectorsOffset = vectorIndexFieldMetadata.getCompressedVectorsOffset();
            this.dimension = vectorIndexFieldMetadata.getVectorDimension();
            this.graphNodeIdToDocMap = vectorIndexFieldMetadata.getGraphNodeIdToDocMap();

            this.vectorIndexFieldDataFileName = baseDataFileName + "_" + fieldInfo.name + "." + JVectorFormat.VECTOR_INDEX_EXTENSION;
            this.neighborsScoreCacheIndexFieldFileName = baseDataFileName
                + "_"
                + fieldInfo.name
                + "."
                + JVectorFormat.NEIGHBORS_SCORE_CACHE_EXTENSION;

            // For the slice we would like to include the Lucene header, unfortunately, we have to do this because jVector use global
            // offsets instead of local offsets
            final long sliceLength = vectorIndexLength + CodecUtil.indexHeaderLength(
                JVectorFormat.VECTOR_INDEX_CODEC_NAME,
                state.segmentSuffix
            );
            // Load the graph index
            this.indexReaderSupplier = new JVectorRandomAccessReader.Supplier(
                directory.openInput(vectorIndexFieldDataFileName, state.context),
                0,
                sliceLength
            );
            this.index = OnDiskGraphIndex.load(indexReaderSupplier, vectorIndexOffset);

            // Load compressed vectors if present
            final byte qType = vectorIndexFieldMetadata.getQuantizationType();
            if (qType == JVectorWriter.QUANTIZATION_TYPE_NVQ_INLINE) {
                log.debug("NVQ vectors stored inline in graph for field {}", fieldInfo.name);
                this.nvqInlineQuantization = extractNVQuantizationFromGraph(this.index);
                if (compressedVectorsLength > 0) {
                    // PQ blob alongside NVQ inline graph — used for approximate search traversal
                    assert compressedVectorsOffset > 0;
                    log.debug("Loading auxiliary PQ blob for NVQ field {}", fieldInfo.name);
                    this.compressedVectorsReaderSupplier = new JVectorRandomAccessReader.Supplier(
                        directory.openInput(vectorIndexFieldDataFileName, IOContext.READONCE),
                        compressedVectorsOffset,
                        compressedVectorsLength
                    );
                    try (final var randomAccessReader = compressedVectorsReaderSupplier.get()) {
                        this.pqVectors = PQVectors.load(randomAccessReader);
                    }
                } else {
                    this.compressedVectorsReaderSupplier = null;
                    this.pqVectors = null;
                }
            } else if (compressedVectorsLength > 0) {
                assert compressedVectorsOffset > 0;
                if (compressedVectorsOffset < vectorIndexOffset) {
                    throw new IllegalArgumentException("compressedVectorsOffset must be greater than vectorIndexOffset");
                }
                log.debug("Loading PQ codebooks and vectors for field {}", fieldInfo.name);
                this.compressedVectorsReaderSupplier = new JVectorRandomAccessReader.Supplier(
                    directory.openInput(vectorIndexFieldDataFileName, IOContext.READONCE),
                    compressedVectorsOffset,
                    compressedVectorsLength
                );
                try (final var randomAccessReader = compressedVectorsReaderSupplier.get()) {
                    this.pqVectors = PQVectors.load(randomAccessReader);
                }
                this.nvqInlineQuantization = null;
            } else {
                this.compressedVectorsReaderSupplier = null;
                this.pqVectors = null;
                this.nvqInlineQuantization = null;
            }

            final IndexInput indexInput = directory.openInput(neighborsScoreCacheIndexFieldFileName, state.context);
            CodecUtil.readIndexHeader(indexInput);

            this.neighborsScoreCacheIndexReaderSupplier = new JVectorRandomAccessReader.Supplier(indexInput);
        }

        /**
         * Extracts the {@link NVQuantization} object from the private {@code nvq} field of
         * the {@link NVQ} feature stored in the given graph index.  This is needed so that
         * the merge path can dequantize NVQ-inline vectors back to float.
         *
         * The jVector {@link NVQ} class does not expose a public getter for its internal
         * {@link NVQuantization}; reflection is the only alternative to modifying the library.
         */
        private static NVQuantization extractNVQuantizationFromGraph(OnDiskGraphIndex index) throws IOException {
            NVQ nvqFeature = (NVQ) index.getFeatures().get(FeatureId.NVQ_VECTORS);
            if (nvqFeature == null) {
                return null;
            }
            try {
                Field nvqField = NVQ.class.getDeclaredField("nvq");
                nvqField.setAccessible(true);
                return (NVQuantization) nvqField.get(nvqFeature);
            } catch (NoSuchFieldException | IllegalAccessException e) {
                throw new IOException("Unable to extract NVQuantization from NVQ feature via reflection", e);
            }
        }

        @Override
        public void close() throws IOException {
            if (indexReaderSupplier != null) {
                IOUtils.close(indexReaderSupplier::close);
            }
            if (compressedVectorsReaderSupplier != null) {
                IOUtils.close(compressedVectorsReaderSupplier::close);
            }
            if (neighborsScoreCacheIndexReaderSupplier != null) {
                IOUtils.close(neighborsScoreCacheIndexReaderSupplier::close);
            }
        }
    }

    /**
     * Utility class to map between Lucene and jVector similarity functions and metadata ordinals.
     */
    public static class VectorSimilarityMapper {
        /**
         List of vector similarity functions supported by <a href="https://github.com/jbellis/jvector">jVector library</a>
         The similarity functions orders matter in this list because it is later used to resolve the similarity function by ordinal.
         */
        public static final List<VectorSimilarityFunction> JVECTOR_SUPPORTED_SIMILARITY_FUNCTIONS = List.of(
            VectorSimilarityFunction.EUCLIDEAN,
            VectorSimilarityFunction.DOT_PRODUCT,
            VectorSimilarityFunction.COSINE,
            VectorSimilarityFunction.DOT_PRODUCT
        );

        public static final Map<org.apache.lucene.index.VectorSimilarityFunction, VectorSimilarityFunction> LUCENE_TO_JVECTOR_MAP = Map.of(
            org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN,
            VectorSimilarityFunction.EUCLIDEAN,
            org.apache.lucene.index.VectorSimilarityFunction.DOT_PRODUCT,
            VectorSimilarityFunction.DOT_PRODUCT,
            org.apache.lucene.index.VectorSimilarityFunction.COSINE,
            VectorSimilarityFunction.COSINE,
            org.apache.lucene.index.VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            VectorSimilarityFunction.DOT_PRODUCT
        );

        public static int distFuncToOrd(org.apache.lucene.index.VectorSimilarityFunction func) {
            if (LUCENE_TO_JVECTOR_MAP.containsKey(func)) {
                return JVECTOR_SUPPORTED_SIMILARITY_FUNCTIONS.indexOf(LUCENE_TO_JVECTOR_MAP.get(func));
            }

            throw new IllegalArgumentException("invalid distance function: " + func);
        }

        public static VectorSimilarityFunction ordToDistFunc(int ord) {
            return JVECTOR_SUPPORTED_SIMILARITY_FUNCTIONS.get(ord);
        }

        public static org.apache.lucene.index.VectorSimilarityFunction ordToLuceneDistFunc(int ord) {
            if (ord < 0 || ord >= JVECTOR_SUPPORTED_SIMILARITY_FUNCTIONS.size()) {
                throw new IllegalArgumentException("Invalid ordinal: " + ord);
            }
            VectorSimilarityFunction jvectorFunc = JVECTOR_SUPPORTED_SIMILARITY_FUNCTIONS.get(ord);
            for (Map.Entry<org.apache.lucene.index.VectorSimilarityFunction, VectorSimilarityFunction> entry : LUCENE_TO_JVECTOR_MAP
                .entrySet()) {
                if (entry.getValue().equals(jvectorFunc)) {
                    return entry.getKey();
                }
            }
            throw new IllegalStateException("No matching Lucene VectorSimilarityFunction found for ordinal: " + ord);
        }
    }
}

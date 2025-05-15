/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.*;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.*;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.io.IOException;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Log4j2
public class JVectorReader extends KnnVectorsReader {
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int DEFAULT_OVER_QUERY_FACTOR = 5; // We will query 5x more than topKFor reranking

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
        String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, JVectorFormat.META_EXTENSION);
        this.directory = state.directory;
        boolean success = false;
        try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
            CodecUtil.checkIndexHeader(
                meta,
                JVectorFormat.META_CODEC_NAME,
                JVectorFormat.VERSION_START,
                JVectorFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            readFields(meta);
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
        // This is already done when loading the fields
        // TODO: Implement this, for now this will always pass
        // CodecUtil.checksumEntireFile(vectorIndex);
        for (FieldEntry fieldEntry : fieldEntryMap.values()) {
            try (var indexInput = state.directory.openInput(fieldEntry.vectorIndexFieldFileName, state.context)) {
                CodecUtil.checksumEntireFile(indexInput);
            }
        }
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        return new JVectorFloatVectorValues(fieldEntryMap.get(field).index, fieldEntryMap.get(field).similarityFunction);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        /**
         * Byte vector values are not supported in jVector library. Instead use PQ.
         */
        return null;
    }

    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        final OnDiskGraphIndex index = fieldEntryMap.get(field).index;

        // search for a random vector using a GraphSearcher and SearchScoreProvider
        VectorFloat<?> q = VECTOR_TYPE_SUPPORT.createFloatVector(target);
        final SearchScoreProvider ssp;

        try (var view = index.getView()) {
            if (fieldEntryMap.get(field).pqVectors != null) { // Quantized, use the precomputed score function
                final PQVectors pqVectors = fieldEntryMap.get(field).pqVectors;
                // SearchScoreProvider that does a first pass with the loaded-in-memory PQVectors,
                // then reranks with the exact vectors that are stored on disk in the index
                ScoreFunction.ApproximateScoreFunction asf = pqVectors.precomputedScoreFunctionFor(
                    q,
                    fieldEntryMap.get(field).similarityFunction
                );
                ScoreFunction.ExactScoreFunction reranker = view.rerankerFor(q, fieldEntryMap.get(field).similarityFunction);
                ssp = new SearchScoreProvider(asf, reranker);
            } else { // Not quantized, used typical searcher
                ssp = SearchScoreProvider.exact(q, fieldEntryMap.get(field).similarityFunction, view);
            }
            io.github.jbellis.jvector.util.Bits compatibleBits = doc -> acceptDocs == null || acceptDocs.get(doc);
            try (var graphSearcher = new GraphSearcher(index)) {
                final var searchResults = graphSearcher.search(
                    ssp,
                    knnCollector.k(),
                    knnCollector.k() * DEFAULT_OVER_QUERY_FACTOR,
                    0.0f,
                    0.0f,
                    compatibleBits
                );
                for (SearchResult.NodeScore ns : searchResults.getNodes()) {
                    knnCollector.collect(ns.node, ns.score);
                }
                // Collect the below metrics about the search and somehow wire this back to {@link @KNNStats}
                final int visitedNodesCount = searchResults.getVisitedCount();
                final int rerankedCount = searchResults.getRerankedCount();

                // TODO: bring back when we re-introduce levels in jVector
                // final int expandedCount = searchResults.getExpandedCount();
                // final int expandedBaseLayerCount = searchResults.getExpandedCountBaseLayer();

                KNNCounter.KNN_QUERY_VISITED_NODES.add(visitedNodesCount);
                KNNCounter.KNN_QUERY_RERANKED_COUNT.add(rerankedCount);
                // TODO: bring back when we re-introduce levels in jVector
                // KNNCounter.KNN_QUERY_EXPANDED_NODES.add(expandedCount);
                // KNNCounter.KNN_QUERY_EXPANDED_BASE_LAYER_NODES.add(expandedBaseLayerCount);

            }
        }
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        // TODO: implement this
    }

    @Override
    public void close() throws IOException {
        for (FieldEntry fieldEntry : fieldEntryMap.values()) {
            IOUtils.close(fieldEntry.readerSupplier::close);
        }
    }

    private void readFields(ChecksumIndexInput meta) throws IOException {
        for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
            final FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldNumber); // read field number
            JVectorWriter.VectorIndexFieldMetadata vectorIndexFieldMetadata = new JVectorWriter.VectorIndexFieldMetadata(meta);
            assert fieldInfo.number == vectorIndexFieldMetadata.getFieldNumber();
            fieldEntryMap.put(fieldInfo.name, new FieldEntry(fieldInfo, vectorIndexFieldMetadata));
        }
    }

    class FieldEntry {
        private final FieldInfo fieldInfo;
        private final VectorEncoding vectorEncoding;
        private final VectorSimilarityFunction similarityFunction;
        private final int dimension;
        private final long vectorIndexOffset;
        private final long vectorIndexLength;
        private final long pqCodebooksAndVectorsLength;
        private final long pqCodebooksAndVectorsOffset;
        private final String vectorIndexFieldFileName;
        private final ReaderSupplier readerSupplier;
        private final OnDiskGraphIndex index;
        private final PQVectors pqVectors; // The product quantized vectors with their codebooks

        public FieldEntry(FieldInfo fieldInfo, JVectorWriter.VectorIndexFieldMetadata vectorIndexFieldMetadata) throws IOException {
            this.fieldInfo = fieldInfo;
            this.similarityFunction = VectorSimilarityMapper.ordToDistFunc(
                vectorIndexFieldMetadata.getVectorSimilarityFunction().ordinal()
            );
            this.vectorEncoding = vectorIndexFieldMetadata.getVectorEncoding();
            this.vectorIndexOffset = vectorIndexFieldMetadata.getVectorIndexOffset();
            this.vectorIndexLength = vectorIndexFieldMetadata.getVectorIndexLength();
            this.pqCodebooksAndVectorsLength = vectorIndexFieldMetadata.getPqCodebooksAndVectorsLength();
            this.pqCodebooksAndVectorsOffset = vectorIndexFieldMetadata.getPqCodebooksAndVectorsOffset();
            this.dimension = vectorIndexFieldMetadata.getVectorDimension();

            this.vectorIndexFieldFileName = baseDataFileName + "_" + fieldInfo.name + "." + JVectorFormat.VECTOR_INDEX_EXTENSION;

            // Check the header
            try (IndexInput indexInput = directory.openInput(vectorIndexFieldFileName, state.context)) {
                CodecUtil.checkIndexHeader(
                    indexInput,
                    JVectorFormat.VECTOR_INDEX_CODEC_NAME,
                    JVectorFormat.VERSION_START,
                    JVectorFormat.VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
                );
            }

            // Load the graph index
            this.readerSupplier = new JVectorRandomAccessReader.Supplier(directory, vectorIndexFieldFileName, state.context);
            this.index = OnDiskGraphIndex.load(readerSupplier, vectorIndexOffset);
            // If quantized load the compressed product quantized vectors with their codebooks
            if (pqCodebooksAndVectorsLength > 0) {
                log.debug(
                    "Loading PQ codebooks and vectors for field {}, with numbers of vectors: {}",
                    fieldInfo.name,
                    state.segmentInfo.maxDoc()
                );
                assert pqCodebooksAndVectorsOffset > 0;
                if (pqCodebooksAndVectorsOffset < vectorIndexOffset) {
                    throw new IllegalArgumentException("pqCodebooksAndVectorsOffset must be greater than vectorIndexOffset");
                }
                try (final var randomAccessReader = readerSupplier.get()) {
                    randomAccessReader.seek(pqCodebooksAndVectorsOffset);
                    this.pqVectors = PQVectors.load(randomAccessReader);
                }
            } else {
                this.pqVectors = null;
            }

            // Check the footer
            try (ChecksumIndexInput indexInput = directory.openChecksumInput(vectorIndexFieldFileName)) {
                // If there are pq codebooks and vectors, then the footer is at the end of the pq codebooks and vectors
                if (pqCodebooksAndVectorsOffset > 0) {
                    indexInput.seek(pqCodebooksAndVectorsOffset + pqCodebooksAndVectorsLength);
                    CodecUtil.checkFooter(indexInput);
                } else {
                    // If there are no pq codebooks and vectors, then the footer is at the end of the vector index
                    // file
                    indexInput.seek(vectorIndexOffset + vectorIndexLength);
                    CodecUtil.checkFooter(indexInput);
                }

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
            VectorSimilarityFunction.COSINE
        );

        public static final Map<org.apache.lucene.index.VectorSimilarityFunction, VectorSimilarityFunction> LUCENE_TO_JVECTOR_MAP = Map.of(
            org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN,
            VectorSimilarityFunction.EUCLIDEAN,
            org.apache.lucene.index.VectorSimilarityFunction.DOT_PRODUCT,
            VectorSimilarityFunction.DOT_PRODUCT,
            org.apache.lucene.index.VectorSimilarityFunction.COSINE,
            VectorSimilarityFunction.COSINE
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

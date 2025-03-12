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
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundDirectory;

import java.io.IOException;
import java.nio.file.Path;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Log4j2
public class JVectorReader extends KnnVectorsReader {
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int DEFAULT_OVER_QUERY_FACTOR = 5; // We will query 5x more than topKFor reranking

    private final FieldInfos fieldInfos;
    private final String indexDataFileName;
    private final String baseDataFileName;
    private final Path directoryBasePath;
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
        this.directoryBasePath = resolveDirectoryPath(directory);
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
            Set<String> filenames = state.segmentInfo.files();
            readFields(meta);
            CodecUtil.checkFooter(meta);

            this.indexDataFileName = IndexFileNames.segmentFileName(
                state.segmentInfo.name,
                state.segmentSuffix,
                JVectorFormat.VECTOR_INDEX_EXTENSION
            );

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

        if (fieldEntryMap.get(field).pqVectors != null) { // Quantized, use the precomputed score function
            final PQVectors pqVectors = fieldEntryMap.get(field).pqVectors;
            // SearchScoreProvider that does a first pass with the loaded-in-memory PQVectors,
            // then reranks with the exact vectors that are stored on disk in the index
            ScoreFunction.ApproximateScoreFunction asf = pqVectors.precomputedScoreFunctionFor(
                q,
                fieldEntryMap.get(field).similarityFunction
            );
            ScoreFunction.ExactScoreFunction reranker = index.getView().rerankerFor(q, fieldEntryMap.get(field).similarityFunction);
            ssp = new SearchScoreProvider(asf, reranker);
        } else { // Not quantized, used typical searcher
            ssp = SearchScoreProvider.exact(q, fieldEntryMap.get(field).similarityFunction, index.getView());
        }
        io.github.jbellis.jvector.util.Bits compatibleBits = doc -> acceptDocs == null || acceptDocs.get(doc);
        final var sr = fieldEntryMap.get(field).graphSearcher.search(ssp, knnCollector.k(), knnCollector.k() * DEFAULT_OVER_QUERY_FACTOR, 0.0f, 0.0f, compatibleBits);
        for (SearchResult.NodeScore ns : sr.getNodes()) {
            knnCollector.collect(ns.node, ns.score);
        }
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        // TODO: implement this
    }

    @Override
    public void close() throws IOException {
        for (FieldEntry fieldEntry : fieldEntryMap.values()) {
            IOUtils.close(fieldEntry.graphSearcher);
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
        private final ReaderSupplier readerSupplier;
        private final OnDiskGraphIndex index;
        private final GraphSearcher graphSearcher;
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
            // TODO: do not depend on the actual nio.Path switch to file name only!
            final Path expectedIndexFilePath = JVectorFormat.getVectorIndexPath(directoryBasePath, baseDataFileName, fieldInfo.name);
            final String originalIndexFileName = expectedIndexFilePath.getFileName().toString();
            final Path indexFilePath;
            final long sliceOffset;
            if (state.segmentInfo.getUseCompoundFile()) {
                if (directory instanceof JVectorCompoundFormat.JVectorCompoundReader) {
                    JVectorCompoundFormat.JVectorCompoundReader jVectorCompoundReader =
                        (JVectorCompoundFormat.JVectorCompoundReader) directory;
                    sliceOffset = jVectorCompoundReader.getOffsetInCompoundFile(originalIndexFileName);
                    indexFilePath = jVectorCompoundReader.getCompoundFilePath();
                } else if (directory instanceof KNN80CompoundDirectory) {
                    KNN80CompoundDirectory knn80CompoundDirectory = (KNN80CompoundDirectory) directory;
                    JVectorCompoundFormat.JVectorCompoundReader jVectorCompoundReader = new JVectorCompoundFormat.JVectorCompoundReader(
                        knn80CompoundDirectory.getDelegate(),
                        knn80CompoundDirectory.getDir(),
                        state.segmentInfo
                    );
                    sliceOffset = jVectorCompoundReader.getOffsetInCompoundFile(originalIndexFileName);
                    indexFilePath = jVectorCompoundReader.getCompoundFilePath();
                } else {
                    throw new IllegalArgumentException(
                        "directory must be JVectorCompoundFormat or KNN80CompoundDirectory but instead had type: "
                            + directory.getClass().getName()
                    );
                }

            } else {
                sliceOffset = 0;
                indexFilePath = expectedIndexFilePath;
            }

            // Check the header
            try (IndexInput indexInput = directory.openInput(originalIndexFileName, state.context)) {
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
            this.readerSupplier = ReaderSupplierFactory.open(indexFilePath);
            this.index = OnDiskGraphIndex.load(readerSupplier, sliceOffset + vectorIndexOffset);
            this.graphSearcher = new GraphSearcher(index);
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
                    randomAccessReader.seek(sliceOffset + pqCodebooksAndVectorsOffset);
                    this.pqVectors = PQVectors.load(randomAccessReader);
                }
            } else {
                this.pqVectors = null;
            }

            // Check the footer
            try (ChecksumIndexInput indexInput = directory.openChecksumInput(originalIndexFileName)) {
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

    public static Path resolveDirectoryPath(Directory dir) {
        while (!(dir instanceof FSDirectory)) {
            final String dirType = dir.getClass().getName();
            log.info("unwrapping dir of type: {} to find path", dirType);
            if (dir instanceof FilterDirectory) {
                dir = ((FilterDirectory) dir).getDelegate();
            } else if (dir instanceof JVectorCompoundFormat.JVectorCompoundReader) {
                return ((JVectorCompoundFormat.JVectorCompoundReader) dir).getDirectoryBasePath();
            } else if (dir instanceof KNN80CompoundDirectory) {
                dir = ((KNN80CompoundDirectory) dir).getDir();
            } else {
                throw new IllegalArgumentException("directory must be FSDirectory or a wrapper around it but instead had type: " + dirType);
            }
        }
        final Path path = ((FSDirectory) dir).getDirectory();
        log.info("resolved directory path from FSDirectory: {}", path);
        return path;
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.*;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Value;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.*;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.time.Clock;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Function;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readVectorEncoding;
import static org.opensearch.knn.index.codec.jvector.JVectorFormat.SIMD_POOL_FLUSH;
import static org.opensearch.knn.index.codec.jvector.JVectorFormat.SIMD_POOL_MERGE;

/**
 * JVectorWriter is responsible for writing vector data into index segments using the JVector library.
 *
 * <h2>Persisting the JVector Graph Index</h2>
 *
 * <p>
 * Flushing data into disk segments occurs in two scenarios:
 * <ol>
 *     <li>When the segment is being flushed to disk (e.g., when a new segment is created) via {@link #flush(int, Sorter.DocMap)}</li>
 *     <li>When the segment is a result of a merge (e.g., when multiple segments are merged into one) via {@link #mergeOneField(FieldInfo, MergeState)}</li>
 * </ol>
 *
 * <h2>jVector Graph Ordinal to Lucene Document ID Mapping</h2>
 *
 * <p>
 * JVector keeps its own ordinals to identify its nodes. Those ordinals can be different from the Lucene document IDs.
 * Document IDs in Lucene can change after a merge operation. Therefore, we need to maintain a mapping between
 * JVector ordinals and Lucene document IDs that can hold across merges.
 * <p>
 * Document IDs in Lucene are mapped across merges and sorts using the {@link org.apache.lucene.index.MergeState.DocMap} for merges and {@link org.apache.lucene.index.Sorter.DocMap} for flush/sorts.
 * For jVector however, we don't want to modify the ordinals in the jVector graph, and therefore we need to maintain a mapping between the jVector ordinals and the new Lucene document IDs.
 * This is achieved by keeping checkpoints of the {@link GraphNodeIdToDocMap} class in the index metadata and allowing us to update the mapping as needed across merges by constructing a new mapping from the previous mapping and the {@link MergeState.DocMap} provided in the {@link MergeState}.
 * And across sorts with {@link GraphNodeIdToDocMap#update(Sorter.DocMap)} during flushes.
 * <p>
 *
 */
@Log4j2
public class JVectorWriter extends KnnVectorsWriter {
    private static final long SHALLOW_RAM_BYTES_USED = RamUsageEstimator.shallowSizeOfInstance(JVectorWriter.class);

    private final List<FieldWriter<?>> fields = new ArrayList<>();

    private final IndexOutput meta;
    private final IndexOutput vectorIndex;
    private final String indexDataFileName;
    private final String baseDataFileName;
    private final SegmentWriteState segmentWriteState;
    private final int maxConn;
    private final int beamWidth;
    private final float degreeOverflow;
    private final float alpha;
    private final Function<Integer, Integer> numberOfSubspacesPerVectorSupplier; // Number of subspaces used per vector for PQ quantization
                                                                                 // as a function of the original dimension
    private final int minimumBatchSizeForQuantization; // Threshold for the vector count above which we will trigger PQ quantization
    private final boolean hierarchyEnabled;

    private boolean finished = false;

    public JVectorWriter(
        SegmentWriteState segmentWriteState,
        int maxConn,
        int beamWidth,
        float degreeOverflow,
        float alpha,
        Function<Integer, Integer> numberOfSubspacesPerVectorSupplier,
        int minimumBatchSizeForQuantization,
        boolean hierarchyEnabled
    ) throws IOException {
        this.segmentWriteState = segmentWriteState;
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.degreeOverflow = degreeOverflow;
        this.alpha = alpha;
        this.numberOfSubspacesPerVectorSupplier = numberOfSubspacesPerVectorSupplier;
        this.minimumBatchSizeForQuantization = minimumBatchSizeForQuantization;
        this.hierarchyEnabled = hierarchyEnabled;
        String metaFileName = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            JVectorFormat.META_EXTENSION
        );

        this.indexDataFileName = IndexFileNames.segmentFileName(
            segmentWriteState.segmentInfo.name,
            segmentWriteState.segmentSuffix,
            JVectorFormat.VECTOR_INDEX_EXTENSION
        );
        this.baseDataFileName = segmentWriteState.segmentInfo.name + "_" + segmentWriteState.segmentSuffix;

        boolean success = false;
        try {
            meta = segmentWriteState.directory.createOutput(metaFileName, segmentWriteState.context);
            vectorIndex = segmentWriteState.directory.createOutput(indexDataFileName, segmentWriteState.context);
            CodecUtil.writeIndexHeader(
                meta,
                JVectorFormat.META_CODEC_NAME,
                JVectorFormat.VERSION_CURRENT,
                segmentWriteState.segmentInfo.getId(),
                segmentWriteState.segmentSuffix
            );

            CodecUtil.writeIndexHeader(
                vectorIndex,
                JVectorFormat.VECTOR_INDEX_CODEC_NAME,
                JVectorFormat.VERSION_CURRENT,
                segmentWriteState.segmentInfo.getId(),
                segmentWriteState.segmentSuffix
            );

            success = true;
        } finally {
            if (!success) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        log.info("Adding field {} in segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name);
        if (fieldInfo.getVectorEncoding() == VectorEncoding.BYTE) {
            final String errorMessage = "byte[] vectors are not supported in JVector. "
                + "Instead you should only use float vectors and leverage product quantization during indexing."
                + "This can provides much greater savings in storage and memory";
            log.error(errorMessage);
            throw new UnsupportedOperationException(errorMessage);
        }
        FieldWriter<?> newField = new FieldWriter<>(fieldInfo, segmentWriteState.segmentInfo.name);

        fields.add(newField);
        return newField;
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        log.info("Merging field {} into segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name);
        try {
            final long mergeStart = Clock.systemDefaultZone().millis();
            switch (fieldInfo.getVectorEncoding()) {
                case BYTE:
                    throw new UnsupportedEncodingException("Byte vectors are not supported in JVector.");
                case FLOAT32:
                    final var mergeRavv = new RandomAccessMergedFloatVectorValues(fieldInfo, mergeState);
                    mergeRavv.merge();
                    break;
            }
            final long mergeEnd = Clock.systemDefaultZone().millis();
            final long mergeTime = mergeEnd - mergeStart;
            KNNCounter.KNN_GRAPH_MERGE_TIME.add(mergeTime);
            log.info("Completed Merge field {} into segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name);
        } catch (Exception e) {
            log.error("Error merging field {} into segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name, e);
            throw e;
        }
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        log.info("Flushing {} fields", fields.size());

        log.info("Flushing jVector graph index");
        for (FieldWriter<?> field : fields) {
            final RandomAccessVectorValues randomAccessVectorValues = field.randomAccessVectorValues;
            final int[] newToOldOrds = new int[randomAccessVectorValues.size()];
            for (int ord = 0; ord < randomAccessVectorValues.size(); ord++) {
                newToOldOrds[ord] = ord;
            }
            final BuildScoreProvider buildScoreProvider;
            final PQVectors pqVectors;
            final FieldInfo fieldInfo = field.fieldInfo;
            if (randomAccessVectorValues.size() >= minimumBatchSizeForQuantization) {
                log.info("Calculating codebooks and compressed vectors for field {}", fieldInfo.name);
                pqVectors = getPQVectors(newToOldOrds, randomAccessVectorValues, fieldInfo);
                buildScoreProvider = BuildScoreProvider.pqBuildScoreProvider(getVectorSimilarityFunction(fieldInfo), pqVectors);
            } else {
                log.info(
                    "Vector count: {}, less than limit to trigger PQ quantization: {}, for field {}, will use full precision vectors instead.",
                    randomAccessVectorValues.size(),
                    minimumBatchSizeForQuantization,
                    fieldInfo.name
                );
                pqVectors = null;
                buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(
                    randomAccessVectorValues,
                    getVectorSimilarityFunction(fieldInfo)
                );
            }

            // Generate the ord to doc mapping
            final int[] ordinalsToDocIds = new int[randomAccessVectorValues.size()];
            for (int ord = 0; ord < randomAccessVectorValues.size(); ord++) {
                ordinalsToDocIds[ord] = field.docIds.get(ord);
            }
            final GraphNodeIdToDocMap graphNodeIdToDocMap = new GraphNodeIdToDocMap(ordinalsToDocIds);
            if (sortMap != null) {
                graphNodeIdToDocMap.update(sortMap);
            }

            OnHeapGraphIndex graph = getGraph(
                buildScoreProvider,
                randomAccessVectorValues,
                newToOldOrds,
                fieldInfo,
                segmentWriteState.segmentInfo.name,
                SIMD_POOL_FLUSH
            );
            writeField(field.fieldInfo, field.randomAccessVectorValues, pqVectors, newToOldOrds, graphNodeIdToDocMap, graph);

        }
    }

    private void writeField(
        FieldInfo fieldInfo,
        RandomAccessVectorValues randomAccessVectorValues,
        PQVectors pqVectors,
        int[] newToOldOrds,
        GraphNodeIdToDocMap graphNodeIdToDocMap,
        OnHeapGraphIndex graph
    ) throws IOException {
        log.info(
            "Writing field {} with vector count: {}, for segment: {}",
            fieldInfo.name,
            randomAccessVectorValues.size(),
            segmentWriteState.segmentInfo.name
        );
        final var vectorIndexFieldMetadata = writeGraph(
            graph,
            randomAccessVectorValues,
            fieldInfo,
            pqVectors,
            newToOldOrds,
            graphNodeIdToDocMap
        );
        meta.writeInt(fieldInfo.number);
        vectorIndexFieldMetadata.toOutput(meta);

        log.info("Writing neighbors score cache for field {}", fieldInfo.name);
        // field data file, which contains the graph
        final String neighborsScoreCacheIndexFieldFileName = baseDataFileName
            + "_"
            + fieldInfo.name
            + "."
            + JVectorFormat.NEIGHBORS_SCORE_CACHE_EXTENSION;
        try (
            IndexOutput indexOutput = segmentWriteState.directory.createOutput(
                neighborsScoreCacheIndexFieldFileName,
                segmentWriteState.context
            );
            final var jVectorIndexWriter = new JVectorIndexWriter(indexOutput)
        ) {
            CodecUtil.writeIndexHeader(
                indexOutput,
                JVectorFormat.NEIGHBORS_SCORE_CACHE_CODEC_NAME,
                JVectorFormat.VERSION_CURRENT,
                segmentWriteState.segmentInfo.getId(),
                segmentWriteState.segmentSuffix
            );
            graph.save(jVectorIndexWriter);
            CodecUtil.writeFooter(indexOutput);
        }
    }

    /**
     * Writes the graph and PQ codebooks and compressed vectors to the vector index file
     * @param graph graph
     * @param randomAccessVectorValues random access vector values
     * @param fieldInfo field info
     * @return Tuple of start offset and length of the graph
     * @throws IOException IOException
     */
    private VectorIndexFieldMetadata writeGraph(
        OnHeapGraphIndex graph,
        RandomAccessVectorValues randomAccessVectorValues,
        FieldInfo fieldInfo,
        PQVectors pqVectors,
        int[] newToOldOrds,
        GraphNodeIdToDocMap graphNodeIdToDocMap
    ) throws IOException {
        // field data file, which contains the graph
        final String vectorIndexFieldFileName = baseDataFileName + "_" + fieldInfo.name + "." + JVectorFormat.VECTOR_INDEX_EXTENSION;

        try (
            IndexOutput indexOutput = segmentWriteState.directory.createOutput(vectorIndexFieldFileName, segmentWriteState.context);
            final var jVectorIndexWriter = new JVectorIndexWriter(indexOutput)
        ) {
            // Header for the field data file
            CodecUtil.writeIndexHeader(
                indexOutput,
                JVectorFormat.VECTOR_INDEX_CODEC_NAME,
                JVectorFormat.VERSION_CURRENT,
                segmentWriteState.segmentInfo.getId(),
                segmentWriteState.segmentSuffix
            );
            final long startOffset = indexOutput.getFilePointer();

            log.info("Writing graph to {}", vectorIndexFieldFileName);
            var resultBuilder = VectorIndexFieldMetadata.builder()
                .fieldNumber(fieldInfo.number)
                .vectorEncoding(fieldInfo.getVectorEncoding())
                .vectorSimilarityFunction(fieldInfo.getVectorSimilarityFunction())
                .vectorDimension(randomAccessVectorValues.dimension())
                .graphNodeIdToDocMap(graphNodeIdToDocMap);

            try (
                var writer = new OnDiskSequentialGraphIndexWriter.Builder(graph, jVectorIndexWriter).with(
                    new InlineVectors(randomAccessVectorValues.dimension())
                ).build()
            ) {
                var suppliers = Feature.singleStateFactory(
                    FeatureId.INLINE_VECTORS,
                    nodeId -> new InlineVectors.State(randomAccessVectorValues.getVector(newToOldOrds[nodeId]))
                );
                writer.write(suppliers);
                long endGraphOffset = jVectorIndexWriter.position();
                resultBuilder.vectorIndexOffset(startOffset);
                resultBuilder.vectorIndexLength(endGraphOffset - startOffset);

                // If PQ is enabled and we have enough vectors, write the PQ codebooks and compressed vectors
                if (pqVectors != null) {
                    log.info(
                        "Writing PQ codebooks and vectors for field {} since the size is {} >= {}",
                        fieldInfo.name,
                        randomAccessVectorValues.size(),
                        minimumBatchSizeForQuantization
                    );
                    resultBuilder.pqCodebooksAndVectorsOffset(endGraphOffset);
                    // write the compressed vectors and codebooks to disk
                    pqVectors.write(jVectorIndexWriter);
                    resultBuilder.pqCodebooksAndVectorsLength(jVectorIndexWriter.position() - endGraphOffset);
                } else {
                    resultBuilder.pqCodebooksAndVectorsOffset(0);
                    resultBuilder.pqCodebooksAndVectorsLength(0);
                }
                CodecUtil.writeFooter(indexOutput);
            }

            return resultBuilder.build();
        }
    }

    private PQVectors getPQVectors(int[] newToOldOrds, RandomAccessVectorValues randomAccessVectorValues, FieldInfo fieldInfo)
        throws IOException {
        final String fieldName = fieldInfo.name;
        final VectorSimilarityFunction vectorSimilarityFunction = fieldInfo.getVectorSimilarityFunction();
        log.info("Computing PQ codebooks for field {} for {} vectors", fieldName, randomAccessVectorValues.size());
        final long start = Clock.systemDefaultZone().millis();
        final var M = numberOfSubspacesPerVectorSupplier.apply(randomAccessVectorValues.dimension());
        final var numberOfClustersPerSubspace = Math.min(256, randomAccessVectorValues.size()); // number of centroids per
        // subspace
        ProductQuantization pq = ProductQuantization.compute(
            randomAccessVectorValues,
            M, // number of subspaces
            numberOfClustersPerSubspace, // number of centroids per subspace
            vectorSimilarityFunction == VectorSimilarityFunction.EUCLIDEAN, // center the dataset
            UNWEIGHTED,
            SIMD_POOL_MERGE,
            ForkJoinPool.commonPool()
        );

        final long end = Clock.systemDefaultZone().millis();
        final long trainingTime = end - start;
        log.info("Computed PQ codebooks for field {}, in {} millis", fieldName, trainingTime);
        KNNCounter.KNN_QUANTIZATION_TRAINING_TIME.add(trainingTime);
        log.info("Encoding and building PQ vectors for field {} for {} vectors", fieldName, randomAccessVectorValues.size());
        // PQVectors pqVectors = pq.encodeAll(randomAccessVectorValues, SIMD_POOL);
        PQVectors pqVectors = PQVectors.encodeAndBuild(pq, newToOldOrds.length, newToOldOrds, randomAccessVectorValues, SIMD_POOL_MERGE);
        log.info(
            "Encoded and built PQ vectors for field {}, original size: {} bytes, compressed size: {} bytes",
            fieldName,
            pqVectors.getOriginalSize(),
            pqVectors.getCompressedSize()
        );
        return pqVectors;
    }

    @Value
    @Builder(toBuilder = true)
    @AllArgsConstructor
    public static class VectorIndexFieldMetadata {
        int fieldNumber;
        VectorEncoding vectorEncoding;
        VectorSimilarityFunction vectorSimilarityFunction;
        int vectorDimension;
        long vectorIndexOffset;
        long vectorIndexLength;
        long pqCodebooksAndVectorsOffset;
        long pqCodebooksAndVectorsLength;
        float degreeOverflow; // important when leveraging cache
        GraphNodeIdToDocMap graphNodeIdToDocMap;

        public void toOutput(IndexOutput out) throws IOException {
            out.writeInt(fieldNumber);
            out.writeInt(vectorEncoding.ordinal());
            out.writeInt(JVectorReader.VectorSimilarityMapper.distFuncToOrd(vectorSimilarityFunction));
            out.writeVInt(vectorDimension);
            out.writeVLong(vectorIndexOffset);
            out.writeVLong(vectorIndexLength);
            out.writeVLong(pqCodebooksAndVectorsOffset);
            out.writeVLong(pqCodebooksAndVectorsLength);
            out.writeInt(Float.floatToIntBits(degreeOverflow));
            graphNodeIdToDocMap.toOutput(out);
        }

        public VectorIndexFieldMetadata(IndexInput in) throws IOException {
            this.fieldNumber = in.readInt();
            this.vectorEncoding = readVectorEncoding(in);
            this.vectorSimilarityFunction = JVectorReader.VectorSimilarityMapper.ordToLuceneDistFunc(in.readInt());
            this.vectorDimension = in.readVInt();
            this.vectorIndexOffset = in.readVLong();
            this.vectorIndexLength = in.readVLong();
            this.pqCodebooksAndVectorsOffset = in.readVLong();
            this.pqCodebooksAndVectorsLength = in.readVLong();
            this.degreeOverflow = Float.intBitsToFloat(in.readInt());
            this.graphNodeIdToDocMap = new GraphNodeIdToDocMap(in);
        }

    }

    @Override
    public void finish() throws IOException {
        log.info("Finishing segment {}", segmentWriteState.segmentInfo.name);
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;

        if (meta != null) {
            // write end of fields marker
            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
        }

        if (vectorIndex != null) {
            CodecUtil.writeFooter(vectorIndex);
        }

    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, vectorIndex);
    }

    @Override
    public long ramBytesUsed() {
        long total = SHALLOW_RAM_BYTES_USED;
        for (FieldWriter<?> field : fields) {
            // the field tracks the delegate field usage
            total += field.ramBytesUsed();
        }
        return total;
    }

    /**
     * The FieldWriter class is responsible for writing vector field data into index segments.
     * It provides functionality to process vector values as those being added, manage memory usage, and build HNSW graph
     * indexing structures for efficient retrieval during search queries.
     *
     * @param <T> The type of vector value to be handled by the writer.
     * This is often specialized to support specific implementations, such as float[] or byte[] vectors.
     */
    static class FieldWriter<T> extends KnnFieldVectorsWriter<T> {
        private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
        private final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(FieldWriter.class);
        @Getter
        private final FieldInfo fieldInfo;
        private int lastDocID = -1;
        private final String segmentName;
        private final RandomAccessVectorValues randomAccessVectorValues;
        // The ordering of docIds matches the ordering of vectors, the index in this list corresponds to the jVector ordinal
        private final List<VectorFloat<?>> vectors = new ArrayList<>();
        private final List<Integer> docIds = new ArrayList<>();

        FieldWriter(FieldInfo fieldInfo, String segmentName) {
            /**
             * For creating a new field from a flat field vectors writer.
             */
            this.randomAccessVectorValues = new ListRandomAccessVectorValues(vectors, fieldInfo.getVectorDimension());
            this.fieldInfo = fieldInfo;
            this.segmentName = segmentName;
        }

        @Override
        public void addValue(int docID, T vectorValue) throws IOException {
            log.trace("Adding value {} to field {} in segment {}", vectorValue, fieldInfo.name, segmentName);
            if (docID == lastDocID) {
                throw new IllegalArgumentException(
                    "VectorValuesField \""
                        + fieldInfo.name
                        + "\" appears more than once in this document (only one value is allowed per field)"
                );
            }
            docIds.add(docID);
            if (vectorValue instanceof float[]) {
                vectors.add(VECTOR_TYPE_SUPPORT.createFloatVector(vectorValue));
            } else if (vectorValue instanceof byte[]) {
                final String errorMessage = "byte[] vectors are not supported in JVector. "
                    + "Instead you should only use float vectors and leverage product quantization during indexing."
                    + "This can provides much greater savings in storage and memory";
                log.error("{}", errorMessage);
                throw new UnsupportedOperationException(errorMessage);
            } else {
                throw new IllegalArgumentException("Unsupported vector type: " + vectorValue.getClass());
            }

            lastDocID = docID;
        }

        @Override
        public T copyValue(T vectorValue) {
            throw new UnsupportedOperationException("copyValue not supported");
        }

        @Override
        public long ramBytesUsed() {
            return SHALLOW_SIZE + (long) vectors.size() * fieldInfo.getVectorDimension() * Float.BYTES;
        }

    }

    static io.github.jbellis.jvector.vector.VectorSimilarityFunction getVectorSimilarityFunction(FieldInfo fieldInfo) {
        log.info("Matching vector similarity function {} for field {}", fieldInfo.getVectorSimilarityFunction(), fieldInfo.name);
        return switch (fieldInfo.getVectorSimilarityFunction()) {
            case EUCLIDEAN -> io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN;
            case COSINE -> io.github.jbellis.jvector.vector.VectorSimilarityFunction.COSINE;
            case DOT_PRODUCT -> io.github.jbellis.jvector.vector.VectorSimilarityFunction.DOT_PRODUCT;
            default -> throw new IllegalArgumentException("Unsupported similarity function: " + fieldInfo.getVectorSimilarityFunction());
        };
    }

    /**
     * Implementation of RandomAccessVectorValues that directly uses the source
     * FloatVectorValues from multiple segments without copying the vectors.
     *
     * Some details about the implementation logic:
     *
     * First, we identify the leading reader, which is the one with the most live vectors.
     * Second, we build a mapping between the ravv ordinals and the reader index and the ordinal in that reader.
     * Third, we build a mapping between the ravv ordinals and the global doc ids.
     *
     * Very important to note that for the leading graph the node Ids need to correspond to their original ravv ordinals in the reader.
     * This is because we are later going to expand that graph with new vectors from the other readers.
     * While the new vectors can be assigned arbitrary node Ids, the leading graph needs to preserve its original node Ids and map them to the original ravv vector ordinals.
     */
    class RandomAccessMergedFloatVectorValues implements RandomAccessVectorValues {
        private static final int READER_ID = 0;
        private static final int READER_ORD = 1;
        private static final int LEADING_READER_IDX = 0;

        private final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

        // Array of sub-readers
        private final KnnVectorsReader[] readers;
        private final JVectorFloatVectorValues[] perReaderFloatVectorValues;

        // Maps the ravv ordinals to the reader index and the ordinal in that reader. This is allowing us to get a unified view of all the
        // vectors in all the readers with a single unified ordinal space.
        private final int[][] ravvOrdToReaderMapping;

        // Total number of vectors
        private final int size;
        // Total number of documents including those without values
        private final int totalDocsCount;

        // Vector dimension
        private final int dimension;
        private final FieldInfo fieldInfo;
        private final MergeState mergeState;
        private final GraphNodeIdToDocMap graphNodeIdToDocMap;
        private final int[] graphNodeIdsToRavvOrds;
        private boolean deletesFound = false;

        /**
         * Creates a random access view over merged float vector values.
         *
         * @param fieldInfo Field info for the vector field
         * @param mergeState Merge state containing readers and doc maps
         */
        public RandomAccessMergedFloatVectorValues(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
            this.totalDocsCount = Math.toIntExact(Arrays.stream(mergeState.maxDocs).asLongStream().sum());
            this.fieldInfo = fieldInfo;
            this.mergeState = mergeState;

            final String fieldName = fieldInfo.name;

            // Count total vectors, collect readers and identify leading reader, collect base ordinals to later be used to build the mapping
            // between global ordinals and global lucene doc ids
            int totalVectorsCount = 0;
            int totalLiveVectorsCount = 0;
            int dimension = 0;
            int tempLeadingReaderIdx = -1;
            int vectorsCountInLeadingReader = -1;
            List<KnnVectorsReader> allReaders = new ArrayList<>();
            final MergeState.DocMap[] docMaps = mergeState.docMaps.clone();
            final Bits[] liveDocs = mergeState.liveDocs.clone();
            final int[] baseOrds = new int[mergeState.knnVectorsReaders.length];
            final int[] deletedOrds = new int[mergeState.knnVectorsReaders.length]; // counts the number of deleted documents in each reader
                                                                                    // that previously had a vector

            // Find the leading reader, count the total number of live vectors, and the base ordinals for each reader
            for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
                FieldInfos fieldInfos = mergeState.fieldInfos[i];
                baseOrds[i] = totalVectorsCount;
                if (MergedVectorValues.hasVectorValues(fieldInfos, fieldName)) {
                    KnnVectorsReader reader = mergeState.knnVectorsReaders[i];
                    if (reader != null) {
                        FloatVectorValues values = reader.getFloatVectorValues(fieldName);
                        if (values != null) {
                            allReaders.add(reader);
                            int vectorCountInReader = values.size();
                            int liveVectorCountInReader = 0;
                            KnnVectorValues.DocIndexIterator it = values.iterator();
                            while (it.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                                if (liveDocs[i] == null || liveDocs[i].get(it.docID())) {
                                    liveVectorCountInReader++;
                                } else {
                                    deletedOrds[i]++;
                                    deletesFound = true;
                                }
                            }
                            if (liveVectorCountInReader >= vectorsCountInLeadingReader) {
                                vectorsCountInLeadingReader = liveVectorCountInReader;
                                tempLeadingReaderIdx = i;
                            }
                            totalVectorsCount += vectorCountInReader;
                            totalLiveVectorsCount += liveVectorCountInReader;
                            dimension = Math.max(dimension, values.dimension());
                        }
                    }
                }
            }

            assert (totalVectorsCount <= totalDocsCount) : "Total number of vectors exceeds the total number of documents";
            assert (totalLiveVectorsCount <= totalVectorsCount) : "Total number of live vectors exceeds the total number of vectors";
            assert (dimension > 0) : "No vectors found for field " + fieldName;

            this.size = totalVectorsCount;
            this.readers = new KnnVectorsReader[allReaders.size()];
            for (int i = 0; i < readers.length; i++) {
                readers[i] = allReaders.get(i);
            }

            // always swap the leading reader to the first position
            // For this part we need to make sure we also swap all the other metadata arrays that are indexed by reader index
            // Such as readers, docMaps, liveDocs, baseOrds, deletedOrds
            if (tempLeadingReaderIdx != 0) {
                final KnnVectorsReader temp = readers[LEADING_READER_IDX];
                readers[LEADING_READER_IDX] = readers[tempLeadingReaderIdx];
                readers[tempLeadingReaderIdx] = temp;
                // also swap the leading doc map to the first position to match the readers
                final MergeState.DocMap tempDocMap = docMaps[LEADING_READER_IDX];
                docMaps[LEADING_READER_IDX] = docMaps[tempLeadingReaderIdx];
                docMaps[tempLeadingReaderIdx] = tempDocMap;
                // swap base ords
                final int tempBaseOrd = baseOrds[LEADING_READER_IDX];
                baseOrds[LEADING_READER_IDX] = baseOrds[tempLeadingReaderIdx];
                baseOrds[tempLeadingReaderIdx] = tempBaseOrd;
            }

            this.perReaderFloatVectorValues = new JVectorFloatVectorValues[readers.length];
            this.dimension = dimension;

            // Build mapping from global ordinal to [readerIndex, readerOrd]
            this.ravvOrdToReaderMapping = new int[totalDocsCount][2];

            int documentsIterated = 0;

            // Will be used to build the new graphNodeIdToDocMap with the new graph node id to docId mapping.
            // This mapping should not be used to access the vectors at any time during construction, but only after the merge is complete
            // and the new segment is created and used by searchers.
            final int[] graphNodeIdToDocIds = new int[totalLiveVectorsCount];
            this.graphNodeIdsToRavvOrds = new int[totalLiveVectorsCount];

            int graphNodeId = 0;
            if (deletesFound) {
                // If there are deletes, we need to build a new graph from scratch and compact the graph node ids
                // TODO: remove this logic once we support incremental graph building with deletes see
                // https://github.com/opensearch-project/opensearch-jvector/issues/171
                for (int readerIdx = 0; readerIdx < readers.length; readerIdx++) {
                    final JVectorFloatVectorValues values = (JVectorFloatVectorValues) readers[readerIdx].getFloatVectorValues(fieldName);
                    perReaderFloatVectorValues[readerIdx] = values;
                    // For each vector in this reader
                    KnnVectorValues.DocIndexIterator it = values.iterator();

                    for (int docId = it.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = it.nextDoc()) {
                        if (docMaps[readerIdx].get(docId) == -1) {
                            log.warn(
                                "Document {} in reader {} is not mapped to a global ordinal from the merge docMaps. Will skip this document for now",
                                docId,
                                readerIdx
                            );
                        } else {
                            // Mapping from ravv ordinals to [readerIndex, readerOrd]
                            // Map graph node id to ravv ordinal
                            // Map graph node id to doc id
                            final int newGlobalDocId = docMaps[readerIdx].get(docId);
                            final int ravvLocalOrd = it.index();
                            final int ravvGlobalOrd = ravvLocalOrd + baseOrds[readerIdx];
                            graphNodeIdToDocIds[graphNodeId] = newGlobalDocId;
                            graphNodeIdsToRavvOrds[graphNodeId] = ravvGlobalOrd;
                            graphNodeId++;
                            ravvOrdToReaderMapping[ravvGlobalOrd][READER_ID] = readerIdx; // Reader index
                            ravvOrdToReaderMapping[ravvGlobalOrd][READER_ORD] = ravvLocalOrd; // Ordinal in reader
                        }

                        documentsIterated++;
                    }
                }
            } else {
                // If there are no deletes, we can reuse the existing graph and simply remap the ravv ordinals to the new global doc ids
                // for the leading reader we must preserve the original node Ids and map them to the corresponding ravv vectors originally
                // used to build the graph
                // This is necessary because we are later going to expand that graph with new vectors from the other readers.
                // The leading reader is ALWAYS the first one in the readers array
                final JVectorFloatVectorValues leadingReaderValues = (JVectorFloatVectorValues) readers[LEADING_READER_IDX]
                    .getFloatVectorValues(fieldName);
                perReaderFloatVectorValues[LEADING_READER_IDX] = leadingReaderValues;
                var leadingReaderIt = leadingReaderValues.iterator();
                for (int docId = leadingReaderIt.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = leadingReaderIt.nextDoc()) {
                    final int newGlobalDocId = docMaps[LEADING_READER_IDX].get(docId);
                    if (newGlobalDocId == -1) {
                        log.warn(
                            "Document {} in reader {} is not mapped to a global ordinal from the merge docMaps. Will skip this document for now",
                            docId,
                            LEADING_READER_IDX
                        );
                    } else {
                        final int ravvLocalOrd = leadingReaderIt.index();
                        final int ravvGlobalOrd = ravvLocalOrd + baseOrds[LEADING_READER_IDX];
                        graphNodeIdToDocIds[ravvLocalOrd] = newGlobalDocId;
                        graphNodeIdsToRavvOrds[ravvLocalOrd] = ravvGlobalOrd;
                        graphNodeId++;
                        ravvOrdToReaderMapping[ravvGlobalOrd][READER_ID] = LEADING_READER_IDX; // Reader index
                        ravvOrdToReaderMapping[ravvGlobalOrd][READER_ORD] = ravvLocalOrd; // Ordinal in reader
                    }

                    documentsIterated++;
                }

                // For the remaining readers we map the graph node id to the ravv ordinal in the order they appear
                for (int readerIdx = 1; readerIdx < readers.length; readerIdx++) {
                    final JVectorFloatVectorValues values = (JVectorFloatVectorValues) readers[readerIdx].getFloatVectorValues(fieldName);
                    perReaderFloatVectorValues[readerIdx] = values;
                    // For each vector in this reader
                    KnnVectorValues.DocIndexIterator it = values.iterator();

                    for (int docId = it.nextDoc(); docId != DocIdSetIterator.NO_MORE_DOCS; docId = it.nextDoc()) {
                        if (docMaps[readerIdx].get(docId) == -1) {
                            log.warn(
                                "Document {} in reader {} is not mapped to a global ordinal from the merge docMaps. Will skip this document for now",
                                docId,
                                readerIdx
                            );
                        } else {
                            // Mapping from ravv ordinals to [readerIndex, readerOrd]
                            // Map graph node id to ravv ordinal
                            // Map graph node id to doc id
                            final int newGlobalDocId = docMaps[readerIdx].get(docId);
                            final int ravvLocalOrd = it.index();
                            final int ravvGlobalOrd = ravvLocalOrd + baseOrds[readerIdx];
                            graphNodeIdToDocIds[graphNodeId] = newGlobalDocId;
                            graphNodeIdsToRavvOrds[graphNodeId] = ravvGlobalOrd;
                            graphNodeId++;
                            ravvOrdToReaderMapping[ravvGlobalOrd][READER_ID] = readerIdx; // Reader index
                            ravvOrdToReaderMapping[ravvGlobalOrd][READER_ORD] = ravvLocalOrd; // Ordinal in reader
                        }

                        documentsIterated++;
                    }
                }
            }

            if (documentsIterated < totalVectorsCount) {
                throw new IllegalStateException(
                    "More documents were expected than what was found in the readers."
                        + "Expected at least number of total vectors: "
                        + totalVectorsCount
                        + " but found only: "
                        + documentsIterated
                        + " documents."
                );
            }

            this.graphNodeIdToDocMap = new GraphNodeIdToDocMap(graphNodeIdToDocIds);
            log.debug("Created RandomAccessMergedFloatVectorValues with {} total vectors from {} readers", size, readers.length);

        }

        /**
         * Merges the float vector values from multiple readers into a unified structure.
         * This process includes handling product quantization (PQ) for vector compression,
         * generating ord-to-doc mappings, and writing the merged index into a new segment file.
         * <p>
         * The method determines if pre-existing product quantization codebooks are available
         * from the leading reader. If available, it refines them using remaining vectors
         * from other readers in the merge. If no pre-existing codebooks are found and
         * the total vector count meets the required minimum threshold, new codebooks
         * and compressed vectors are computed. Otherwise, no PQ compression is applied.
         * <p>
         * Also, it generates a mapping of ordinals to document IDs by iterating through
         * the provided vector data, which is further used to write the field data.
         * <p>
         * In the event of no deletes or quantization, the graph construction is done by incrementally adding vectors from smaller segments into the largest segment.
         * For all other cases, we build a new graph from scratch from all the vectors.
         *
         * TODO: Add support for incremental graph building with quantization see <a href="https://github.com/opensearch-project/opensearch-jvector/issues/166">issue</a>
         *
         * @throws IOException if there is an issue during reading or writing vector data.
         */
        public void merge() throws IOException {
            // This section creates the PQVectors to be used for this merge
            // Get PQ compressor for leading reader
            final int totalVectorsCount = size;
            final String fieldName = fieldInfo.name;
            final PQVectors pqVectors;
            final OnHeapGraphIndex graph;
            // Get the leading reader
            PerFieldKnnVectorsFormat.FieldsReader fieldsReader = (PerFieldKnnVectorsFormat.FieldsReader) readers[LEADING_READER_IDX];
            JVectorReader leadingReader = (JVectorReader) fieldsReader.getFieldReader(fieldName);
            final BuildScoreProvider buildScoreProvider;
            // Check if the leading reader has pre-existing PQ codebooks and if so, refine them with the remaining vectors
            if (leadingReader.getProductQuantizationForField(fieldInfo.name).isEmpty()) {
                // No pre-existing codebooks, check if we have enough vectors to trigger quantization
                log.info(
                    "No Pre-existing PQ codebooks found in this merge for field {} in segment {}, will check if a new codebooks is necessary",
                    fieldName,
                    mergeState.segmentInfo.name
                );
                if (this.size() >= minimumBatchSizeForQuantization) {
                    log.info(
                        "Calculating new codebooks and compressed vectors for field: {}, with totalVectorCount: {}, above minimumBatchSizeForQuantization: {}",
                        fieldName,
                        totalVectorsCount,
                        minimumBatchSizeForQuantization
                    );
                    pqVectors = getPQVectors(graphNodeIdsToRavvOrds, this, fieldInfo);
                } else {
                    log.info(
                        "Not enough vectors found for field: {}, totalVectorCount: {}, is below minimumBatchSizeForQuantization: {}",
                        fieldName,
                        totalVectorsCount,
                        minimumBatchSizeForQuantization
                    );
                    pqVectors = null;
                }
            } else {
                log.info(
                    "Pre-existing PQ codebooks found in this merge for field {} in segment {}, will refine the codebooks from the leading reader with the remaining vectors",
                    fieldName,
                    mergeState.segmentInfo.name
                );
                final long start = Clock.systemDefaultZone().millis();
                ProductQuantization leadingCompressor = leadingReader.getProductQuantizationForField(fieldName).get();
                // Refine the leadingCompressor with the remaining vectors in the merge, we skip the leading reader since it's already been
                // used to create the leadingCompressor
                // We assume the leading reader is ALWAYS the first one in the readers array
                for (int i = LEADING_READER_IDX + 1; i < readers.length; i++) {
                    final FloatVectorValues values = readers[i].getFloatVectorValues(fieldName);
                    final RandomAccessVectorValues randomAccessVectorValues = new RandomAccessVectorValuesOverVectorValues(values);
                    leadingCompressor.refine(randomAccessVectorValues);
                }
                final long end = Clock.systemDefaultZone().millis();
                final long trainingTime = end - start;
                log.info("Refined PQ codebooks for field {}, in {} millis", fieldName, trainingTime);
                KNNCounter.KNN_QUANTIZATION_TRAINING_TIME.add(trainingTime);
                pqVectors = PQVectors.encodeAndBuild(
                    leadingCompressor,
                    graphNodeIdsToRavvOrds.length,
                    graphNodeIdsToRavvOrds,
                    this,
                    SIMD_POOL_MERGE
                );
            }

            if (pqVectors == null) {
                buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(
                    this,
                    graphNodeIdsToRavvOrds,
                    getVectorSimilarityFunction(fieldInfo)
                );
                // graph = getGraph(buildScoreProvider, this, newToOldOrds, fieldInfo, segmentWriteState.segmentInfo.name);
                if (!deletesFound) {
                    final String segmentName = segmentWriteState.segmentInfo.name;
                    log.info(
                        "No deletes found, and no PQ codebooks found, expanding previous graph with additional vectors for field {} in segment {}",
                        fieldName,
                        segmentName
                    );
                    final RandomAccessReader leadingOnHeapGraphReader = leadingReader.getNeighborsScoreCacheForField(fieldName);
                    final int numBaseVectors = leadingReader.getFloatVectorValues(fieldName).size();
                    graph = (OnHeapGraphIndex) GraphIndexBuilder.buildAndMergeNewNodes(
                        leadingOnHeapGraphReader,
                        this,
                        buildScoreProvider,
                        numBaseVectors,
                        graphNodeIdsToRavvOrds,
                        beamWidth,
                        degreeOverflow,
                        alpha,
                        hierarchyEnabled
                    );
                } else {
                    log.info("Deletes found, and no PQ codebooks found, building new graph from scratch");
                    graph = getGraph(
                        buildScoreProvider,
                        this,
                        graphNodeIdsToRavvOrds,
                        fieldInfo,
                        segmentWriteState.segmentInfo.name,
                        SIMD_POOL_MERGE
                    );
                }
            } else {
                log.info("PQ codebooks found, building graph from scratch with PQ vectors");
                buildScoreProvider = BuildScoreProvider.pqBuildScoreProvider(getVectorSimilarityFunction(fieldInfo), pqVectors);
                // Pre-init the diversity provider here to avoid doing it lazily (as it could block the SIMD threads)
                buildScoreProvider.diversityProviderFor(0);
                graph = getGraph(
                    buildScoreProvider,
                    this,
                    graphNodeIdsToRavvOrds,
                    fieldInfo,
                    segmentWriteState.segmentInfo.name,
                    SIMD_POOL_MERGE
                );
            }

            writeField(fieldInfo, this, pqVectors, graphNodeIdsToRavvOrds, graphNodeIdToDocMap, graph);
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public VectorFloat<?> getVector(int ord) {
            if (ord < 0 || ord >= totalDocsCount) {
                throw new IllegalArgumentException("Ordinal out of bounds: " + ord);
            }

            final int readerIdx = ravvOrdToReaderMapping[ord][READER_ID];
            final int readerOrd = ravvOrdToReaderMapping[ord][READER_ORD];

            // Access to float values is not thread safe
            synchronized (perReaderFloatVectorValues[readerIdx]) {
                return perReaderFloatVectorValues[readerIdx].vectorFloatValue(readerOrd);
            }
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            throw new UnsupportedOperationException("Copy not supported");
        }
    }

    /**
     * This method will return the graph index for the field
     * @return OnHeapGraphIndex
     */
    public OnHeapGraphIndex getGraph(
        BuildScoreProvider buildScoreProvider,
        RandomAccessVectorValues randomAccessVectorValues,
        int[] newToOldOrds,
        FieldInfo fieldInfo,
        String segmentName,
        ForkJoinPool SIMD_POOL
    ) {
        final GraphIndexBuilder graphIndexBuilder = new GraphIndexBuilder(
            buildScoreProvider,
            fieldInfo.getVectorDimension(),
            maxConn,
            beamWidth,
            degreeOverflow,
            alpha,
            hierarchyEnabled
        );

        /*
         * We cannot always use randomAccessVectorValues for the graph building
         * because it's size will not always correspond to the document count.
         * To have the right mapping from docId to vector ordinal we need to use the mergedFloatVector.
         * This is the case when we are merging segments and we might have more documents than vectors.
         */
        final long start = Clock.systemDefaultZone().millis();
        final OnHeapGraphIndex graphIndex;
        var vv = randomAccessVectorValues.threadLocalSupplier();

        log.info("Building graph from merged float vector");
        // parallel graph construction from the merge documents Ids
        SIMD_POOL.submit(() -> IntStream.range(0, newToOldOrds.length).parallel().forEach(ord -> {
            graphIndexBuilder.addGraphNode(ord, vv.get().getVector(newToOldOrds[ord]));
        })).join();
        graphIndexBuilder.cleanup();
        graphIndex = (OnHeapGraphIndex) graphIndexBuilder.getGraph();
        final long end = Clock.systemDefaultZone().millis();

        log.info("Built graph for field {} in segment {} in {} millis", fieldInfo.name, segmentName, end - start);
        return graphIndex;
    }

    static class RandomAccessVectorValuesOverVectorValues implements RandomAccessVectorValues {
        private final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
        private final FloatVectorValues values;

        public RandomAccessVectorValuesOverVectorValues(FloatVectorValues values) {
            this.values = values;
        }

        @Override
        public int size() {
            return values.size();
        }

        @Override
        public int dimension() {
            return values.dimension();
        }

        @Override
        public VectorFloat<?> getVector(int nodeId) {
            try {
                // Access to float values is not thread safe
                synchronized (this) {
                    final float[] vector = values.vectorValue(nodeId);
                    final float[] copy = new float[vector.length];
                    System.arraycopy(vector, 0, copy, 0, vector.length);
                    return VECTOR_TYPE_SUPPORT.createFloatVector(copy);
                }
            } catch (IOException e) {
                log.error("Error retrieving vector at ordinal {}", nodeId, e);
                throw new RuntimeException(e);
            }
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            throw new UnsupportedOperationException("Copy not supported");
        }
    }

}

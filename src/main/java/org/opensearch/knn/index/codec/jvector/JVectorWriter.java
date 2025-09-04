/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
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
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.plugin.stats.KNNCounter;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.time.Clock;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Function;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readVectorEncoding;
import static org.opensearch.knn.index.codec.jvector.JVectorFormat.SIMD_POOL;

@Log4j2
public class JVectorWriter extends KnnVectorsWriter {
    private static final long SHALLOW_RAM_BYTES_USED = RamUsageEstimator.shallowSizeOfInstance(JVectorWriter.class);
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

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
    private final boolean mergeOnDisk;
    private boolean finished = false;

    public JVectorWriter(
        SegmentWriteState segmentWriteState,
        int maxConn,
        int beamWidth,
        float degreeOverflow,
        float alpha,
        Function<Integer, Integer> numberOfSubspacesPerVectorSupplier,
        int minimumBatchSizeForQuantization,
        boolean mergeOnDisk
    ) throws IOException {
        this.segmentWriteState = segmentWriteState;
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.degreeOverflow = degreeOverflow;
        this.alpha = alpha;
        this.numberOfSubspacesPerVectorSupplier = numberOfSubspacesPerVectorSupplier;
        this.minimumBatchSizeForQuantization = minimumBatchSizeForQuantization;
        this.mergeOnDisk = mergeOnDisk;
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
        var success = false;
        try {
            final long mergeStart = Clock.systemDefaultZone().millis();
            switch (fieldInfo.getVectorEncoding()) {
                case BYTE:
                    throw new UnsupportedEncodingException("Byte vectors are not supported in JVector.");
                case FLOAT32:
                    final FieldWriter<float[]> floatVectorFieldWriter;
                    FloatVectorValues mergeFloatVector = MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
                    if (mergeOnDisk) {
                        final var mergeRavv = new RandomAccessMergedFloatVectorValues(fieldInfo, mergeState, mergeFloatVector);
                        mergeRavv.merge();
                    } else {
                        floatVectorFieldWriter = (FieldWriter<float[]>) addField(fieldInfo);
                        var itr = mergeFloatVector.iterator();
                        final List<Integer> docIds = new ArrayList<>(mergeFloatVector.size());
                        for (int doc = itr.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = itr.nextDoc()) {
                            floatVectorFieldWriter.addValue(doc, mergeFloatVector.vectorValue(doc));
                            docIds.add(doc);
                        }
                        final PQVectors pqVectors = getPQVectors(floatVectorFieldWriter.randomAccessVectorValues, fieldInfo);

                        writeField(fieldInfo, floatVectorFieldWriter.randomAccessVectorValues, docIds, pqVectors);
                    }
                    break;
            }
            final long mergeEnd = Clock.systemDefaultZone().millis();
            final long mergeTime = mergeEnd - mergeStart;
            KNNCounter.KNN_GRAPH_MERGE_TIME.add(mergeTime);
            success = true;
            log.info("Completed Merge field {} into segment {}", fieldInfo.name, segmentWriteState.segmentInfo.name);
        } finally {

        }
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        log.info("Flushing {} fields", fields.size());

        log.info("Flushing jVector graph index");
        for (FieldWriter<?> field : fields) {
            final RandomAccessVectorValues randomAccessVectorValues = field.randomAccessVectorValues;
            final PQVectors pqVectors;
            final FieldInfo fieldInfo = field.fieldInfo;
            if (randomAccessVectorValues.size() >= minimumBatchSizeForQuantization) {
                log.info("Calculating codebooks and compressed vectors for field {}", fieldInfo.name);
                pqVectors = getPQVectors(randomAccessVectorValues, fieldInfo);
            } else {
                log.info(
                    "Vector count: {}, less than limit to trigger PQ quantization: {}, for field {}, will use full precision vectors instead.",
                    randomAccessVectorValues.size(),
                    minimumBatchSizeForQuantization,
                    fieldInfo.name
                );
                pqVectors = null;
            }
            if (sortMap == null) {
                final List<Integer> docIds = new ArrayList<>(field.randomAccessVectorValues.size());
                for (int doc = 0; doc < field.randomAccessVectorValues.size(); doc++) {
                    docIds.add(doc);
                }
                writeField(field.fieldInfo, field.randomAccessVectorValues, docIds, pqVectors);
            } else {
                throw new UnsupportedOperationException("Not implemented yet");
                // writeSortingField(field, sortMap);
            }
        }
    }

    private void writeField(
        FieldInfo fieldInfo,
        RandomAccessVectorValues randomAccessVectorValues,
        List<Integer> docIds,
        PQVectors pqVectors
    ) throws IOException {
        log.info(
            "Writing field {} with vector count: {}, for segment: {}",
            fieldInfo.name,
            randomAccessVectorValues.size(),
            segmentWriteState.segmentInfo.name
        );
        final BuildScoreProvider buildScoreProvider;
        if (pqVectors == null) {
            buildScoreProvider = BuildScoreProvider.randomAccessScoreProvider(
                randomAccessVectorValues,
                getVectorSimilarityFunction(fieldInfo)
            );
        } else {
            buildScoreProvider = BuildScoreProvider.pqBuildScoreProvider(getVectorSimilarityFunction(fieldInfo), pqVectors);
        }

        // If we haven't provided ord to docId map we will assume will just generate one based on the ordering of the vectors in the
        // RandomAccessVectorValues
        if (docIds == null) {
            docIds = new ArrayList<>();
            for (int i = 0; i < randomAccessVectorValues.size(); i++) {
                docIds.add(i);
            }
        }
        OnHeapGraphIndex graph = getGraph(
            buildScoreProvider,
            randomAccessVectorValues,
            docIds,
            fieldInfo,
            segmentWriteState.segmentInfo.name
        );
        final var vectorIndexFieldMetadata = writeGraph(graph, randomAccessVectorValues, fieldInfo, pqVectors);
        meta.writeInt(fieldInfo.number);
        vectorIndexFieldMetadata.toOutput(meta);
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
        PQVectors pqVectors
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
                .vectorDimension(randomAccessVectorValues.dimension());

            try (
                var writer = new OnDiskSequentialGraphIndexWriter.Builder(graph, jVectorIndexWriter).with(
                    new InlineVectors(randomAccessVectorValues.dimension())
                ).build()
            ) {
                var suppliers = Feature.singleStateFactory(
                    FeatureId.INLINE_VECTORS,
                    nodeId -> new InlineVectors.State(randomAccessVectorValues.getVector(nodeId))
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

    private PQVectors getPQVectors(RandomAccessVectorValues randomAccessVectorValues, FieldInfo fieldInfo) throws IOException {
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
            SIMD_POOL,
            ForkJoinPool.commonPool()
        );

        final long end = Clock.systemDefaultZone().millis();
        final long trainingTime = end - start;
        log.info("Computed PQ codebooks for field {}, in {} millis", fieldName, trainingTime);
        KNNCounter.KNN_QUANTIZATION_TRAINING_TIME.add(trainingTime);
        log.info("Encoding and building PQ vectors for field {} for {} vectors", fieldName, randomAccessVectorValues.size());
        PQVectors pqVectors = (PQVectors) pq.encodeAll(randomAccessVectorValues, SIMD_POOL);
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

        public void toOutput(IndexOutput out) throws IOException {
            out.writeInt(fieldNumber);
            out.writeInt(vectorEncoding.ordinal());
            out.writeInt(JVectorReader.VectorSimilarityMapper.distFuncToOrd(vectorSimilarityFunction));
            out.writeVInt(vectorDimension);
            out.writeVLong(vectorIndexOffset);
            out.writeVLong(vectorIndexLength);
            out.writeVLong(pqCodebooksAndVectorsOffset);
            out.writeVLong(pqCodebooksAndVectorsLength);
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
    class FieldWriter<T> extends KnnFieldVectorsWriter<T> {
        private final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(FieldWriter.class);
        @Getter
        private final FieldInfo fieldInfo;
        private int lastDocID = -1;
        private final String segmentName;
        private final RandomAccessVectorValues randomAccessVectorValues;
        private final List<VectorFloat<?>> flatVectors;

        FieldWriter(FieldInfo fieldInfo, String segmentName) {
            /**
             * For creating a new field from a flat field vectors writer.
             */
            this.flatVectors = new ArrayList<>();
            this.randomAccessVectorValues = new RandomAccessVectorValuesOverFlatFields(flatVectors, fieldInfo);
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
            if (vectorValue instanceof float[]) {
                flatVectors.add(JVectorWriter.VECTOR_TYPE_SUPPORT.createFloatVector(vectorValue));
            } else if (vectorValue instanceof byte[]) {
                final String errorMessage = "byte[] vectors are not supported in JVector. "
                    + "Instead you should only use float vectors and leverage product quantization during indexing."
                    + "This can provides much greater savings in storage and memory";
                log.error(errorMessage);
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
            return SHALLOW_SIZE + (flatVectors.isEmpty()
                ? 0
                : +(long) flatVectors.size() * flatVectors.getFirst().ramBytesUsed() + (long) flatVectors.size());
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
     */
    class RandomAccessMergedFloatVectorValues implements RandomAccessVectorValues {
        private static final int READER_ID = 0;
        private static final int READER_ORD = 1;

        private final FloatVectorValues mergedFlatFloatVectors;

        // Array of sub-readers
        private final KnnVectorsReader[] readers;
        private final FloatVectorValues[] perReaderFloatVectorValues;

        private final int leadingReaderId;

        // For each ordinal, stores which reader and which ordinal in that reader
        private final int[][] ordMapping;

        // Total number of vectors
        private final int size;
        // Total number of documents including those without values
        private final int totalDocsCount;

        // Vector dimension
        private final int dimension;
        private final FieldInfo fieldInfo;
        private final MergeState mergeState;

        /**
         * Creates a random access view over merged float vector values.
         *
         * @param fieldInfo Field info for the vector field
         * @param mergeState Merge state containing readers and doc maps
         */
        public RandomAccessMergedFloatVectorValues(FieldInfo fieldInfo, MergeState mergeState, FloatVectorValues mergedFlatFloatVectors)
            throws IOException {
            this.totalDocsCount = Math.toIntExact(Arrays.stream(mergeState.maxDocs).asLongStream().sum());
            this.fieldInfo = fieldInfo;
            this.mergeState = mergeState;
            this.mergedFlatFloatVectors = mergedFlatFloatVectors;

            final String fieldName = fieldInfo.name;

            // Count total vectors and collect readers
            int totalVectorsCount = 0;
            int dimension = 0;
            int tempLeadingReaderId = -1;
            int vectorsCountInLeadingReader = -1;
            List<KnnVectorsReader> allReaders = new ArrayList<>();

            for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
                FieldInfos fieldInfos = mergeState.fieldInfos[i];
                if (MergedVectorValues.hasVectorValues(fieldInfos, fieldName)) {
                    KnnVectorsReader reader = mergeState.knnVectorsReaders[i];
                    if (reader != null) {
                        FloatVectorValues values = reader.getFloatVectorValues(fieldName);
                        if (values != null) {
                            allReaders.add(reader);
                            final int vectorCountInReader = values.size();
                            if (vectorCountInReader >= vectorsCountInLeadingReader) {
                                vectorsCountInLeadingReader = vectorCountInReader;
                                tempLeadingReaderId = i;
                            }
                            totalVectorsCount += vectorCountInReader;
                            dimension = Math.max(dimension, values.dimension());
                        }
                    }
                }
            }
            this.leadingReaderId = tempLeadingReaderId;

            assert (totalVectorsCount <= totalDocsCount) : "Total number of vectors exceeds the total number of documents";
            assert (dimension > 0) : "No vectors found for field " + fieldName;

            this.size = totalVectorsCount;
            this.readers = new KnnVectorsReader[allReaders.size()];
            for (int i = 0; i < readers.length; i++) {
                readers[i] = allReaders.get(i);
            }
            this.perReaderFloatVectorValues = new FloatVectorValues[readers.length];
            this.dimension = dimension;

            // Build mapping from global ordinal to [readerIndex, readerOrd]
            this.ordMapping = new int[totalDocsCount][2];

            int documentsIterated = 0;

            // Simulate the merge process to build the ordinal mapping
            // This is similar to what DocIDMerger would do but tracks ordinals
            MergeState.DocMap[] docMaps = mergeState.docMaps;
            // For each reader
            for (int readerIdx = 0; readerIdx < readers.length; readerIdx++) {
                final FloatVectorValues values = readers[readerIdx].getFloatVectorValues(fieldName);
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
                        // Mapping from global ordinal to [readerIndex, readerOrd]
                        final int globalOrd = docMaps[readerIdx].get(docId);
                        ordMapping[globalOrd][READER_ID] = readerIdx; // Reader index
                        ordMapping[globalOrd][READER_ORD] = docId; // Ordinal in reader
                    }

                    documentsIterated++;
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
         *
         * @throws IOException if there is an issue during reading or writing vector data.
         */
        public void merge() throws IOException {
            // This section creates the PQVectors to be used for this merge
            // Get PQ compressor for leading reader
            final int totalVectorsCount = size;
            final String fieldName = fieldInfo.name;
            final PQVectors pqVectors;
            PerFieldKnnVectorsFormat.FieldsReader fieldsReader = (PerFieldKnnVectorsFormat.FieldsReader) readers[leadingReaderId];
            JVectorReader leadingReader = (JVectorReader) fieldsReader.getFieldReader(fieldName);
            if (leadingReader.getProductQuantizationForField(fieldInfo.name).isEmpty()) {
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
                    pqVectors = getPQVectors(this, fieldInfo);
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
                // Refine the leadingCompressor with the remaining vectors in the merge
                for (int i = 0; i < readers.length; i++) {
                    // Avoid recomputing with the leading reader vectors for the tuning, we only want vectors that haven't been used yet
                    if (i != leadingReaderId) {
                        final FloatVectorValues values = readers[i].getFloatVectorValues(fieldName);
                        final RandomAccessVectorValues randomAccessVectorValues = new RandomAccessVectorValuesOverVectorValues(values);
                        leadingCompressor.refine(randomAccessVectorValues);
                    }
                }
                final long end = Clock.systemDefaultZone().millis();
                final long trainingTime = end - start;
                log.info("Refined PQ codebooks for field {}, in {} millis", fieldName, trainingTime);
                KNNCounter.KNN_QUANTIZATION_TRAINING_TIME.add(trainingTime);
                pqVectors = leadingCompressor.encodeAll(this, SIMD_POOL);
            }

            // Generate the ord to doc mapping
            final List<Integer> docIds = new ArrayList<>(totalVectorsCount);
            final KnnVectorValues.DocIndexIterator itr = mergedFlatFloatVectors.iterator();
            while (itr.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                docIds.add(itr.docID());
            }
            writeField(fieldInfo, this, docIds, pqVectors);
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

            final int readerIdx = ordMapping[ord][READER_ID];
            final int readerOrd = ordMapping[ord][READER_ORD];

            // Access to float values is not thread safe
            synchronized (this) {
                final JVectorFloatVectorValues values = (JVectorFloatVectorValues) perReaderFloatVectorValues[readerIdx];
                return values.vectorFloatValue(readerOrd);
            }
        }

        @Override
        public boolean isValueShared() {
            return true;
        }

        @Override
        public RandomAccessVectorValues copy() {
            return this;
        }
    }

    /**
     * This method will return the graph index for the field
     * @return OnHeapGraphIndex
     * @throws IOException IOException
     */
    public OnHeapGraphIndex getGraph(
        BuildScoreProvider buildScoreProvider,
        RandomAccessVectorValues randomAccessVectorValues,
        List<Integer> docIds,
        FieldInfo fieldInfo,
        String segmentName
    ) throws IOException {
        final GraphIndexBuilder graphIndexBuilder = new GraphIndexBuilder(
            buildScoreProvider,
            fieldInfo.getVectorDimension(),
            maxConn,
            beamWidth,
            degreeOverflow,
            alpha,
            true
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
        SIMD_POOL.submit(
            () -> docIds.stream().parallel().forEach(node -> { graphIndexBuilder.addGraphNode(node, vv.get().getVector(node)); })
        ).join();
        graphIndexBuilder.cleanup();
        graphIndex = graphIndexBuilder.getGraph();
        final long end = Clock.systemDefaultZone().millis();

        log.info("Built graph for field {} in segment {} in {} millis", fieldInfo.name, segmentName, end - start);
        return graphIndex;
    }

    static class RandomAccessVectorValuesOverFlatFields implements RandomAccessVectorValues {

        private final List<VectorFloat<?>> flatVectors;
        private final int dimension;

        RandomAccessVectorValuesOverFlatFields(List<VectorFloat<?>> flatVectors, FieldInfo fieldInfo) {
            this.flatVectors = flatVectors;
            this.dimension = fieldInfo.getVectorDimension();
        }

        @Override
        public int size() {
            return flatVectors.size();
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public VectorFloat<?> getVector(int nodeId) {
            return flatVectors.get(nodeId);
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            // only used internally
            return this;
        }
    }

    static class RandomAccessVectorValuesOverVectorValues implements RandomAccessVectorValues {
        private final JVectorFloatVectorValues values;

        public RandomAccessVectorValuesOverVectorValues(FloatVectorValues values) {
            this.values = (JVectorFloatVectorValues) values;
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
            // Access to float values is not thread safe
            synchronized (this) {
                return values.vectorFloatValue(nodeId);
            }
        }

        @Override
        public boolean isValueShared() {
            return true;
        }

        @Override
        public RandomAccessVectorValues copy() {
            // Only used internally
            return this;
        }
    }
}

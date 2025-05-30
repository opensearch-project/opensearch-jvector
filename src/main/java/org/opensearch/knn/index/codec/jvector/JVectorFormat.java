/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

import java.io.IOException;

public class JVectorFormat extends KnnVectorsFormat {
    public static final String NAME = "JVectorFormat";
    public static final String META_CODEC_NAME = "JVectorVectorsFormatMeta";
    public static final String VECTOR_INDEX_CODEC_NAME = "JVectorVectorsFormatIndex";
    public static final String JVECTOR_FILES_SUFFIX = "jvector";
    public static final String META_EXTENSION = "meta-" + JVECTOR_FILES_SUFFIX;
    public static final String VECTOR_INDEX_EXTENSION = "data-" + JVECTOR_FILES_SUFFIX;
    public static final int DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION = 1024; // The minimum number of vectors required to trigger
                                                                                // quantization
    public static final int VERSION_START = 0;
    public static final int VERSION_CURRENT = VERSION_START;
    private static final int DEFAULT_MAX_CONN = 32;
    private static final int DEFAULT_BEAM_WIDTH = 100;
    private static final float DEFAULT_DEGREE_OVERFLOW = 1.2f;
    private static final float DEFAULT_ALPHA = 1.2f;
    public static final boolean DEFAULT_MERGE_ON_DISK = true;

    private final int maxConn;
    private final int beamWidth;
    private final int minBatchSizeForQuantization;
    private final boolean mergeOnDisk;

    public JVectorFormat() {
        this(NAME, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH, DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION, DEFAULT_MERGE_ON_DISK);
    }

    public JVectorFormat(int minBatchSizeForQuantization, boolean mergeOnDisk) {
        this(NAME, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH, minBatchSizeForQuantization, mergeOnDisk);
    }

    public JVectorFormat(int maxConn, int beamWidth, int minBatchSizeForQuantization, boolean mergeOnDisk) {
        this(NAME, maxConn, beamWidth, minBatchSizeForQuantization, mergeOnDisk);
    }

    public JVectorFormat(String name, int maxConn, int beamWidth, int minBatchSizeForQuantization, boolean mergeOnDisk) {
        super(name);
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.minBatchSizeForQuantization = minBatchSizeForQuantization;
        this.mergeOnDisk = mergeOnDisk;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new JVectorWriter(
            state,
            maxConn,
            beamWidth,
            DEFAULT_DEGREE_OVERFLOW,
            DEFAULT_ALPHA,
            minBatchSizeForQuantization,
            mergeOnDisk
        );
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new JVectorReader(state, mergeOnDisk);
    }

    @Override
    public int getMaxDimensions(String s) {
        // Not a hard limit, but a reasonable default
        return 8192;
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;

public class JVectorCodec extends FilterCodec {

    public static final String CODEC_NAME = "JVectorCodec";
    private int minBatchSizeForQuantization;
    private boolean mergeOnDisk;

    public JVectorCodec() {
        this(
            CODEC_NAME,
            new Lucene101Codec(),
            JVectorFormat.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION,
            JVectorFormat.DEFAULT_MERGE_ON_DISK
        );
    }

    public JVectorCodec(int minBatchSizeForQuantization, boolean mergeOnDisk) {
        this(CODEC_NAME, new Lucene101Codec(), minBatchSizeForQuantization, mergeOnDisk);
    }

    public JVectorCodec(String codecName, Codec delegate, int minBatchSizeForQuantization, boolean mergeOnDisk) {
        super(codecName, delegate);
        this.minBatchSizeForQuantization = minBatchSizeForQuantization;
        this.mergeOnDisk = mergeOnDisk;
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return new JVectorFormat(minBatchSizeForQuantization, mergeOnDisk);
    }
}

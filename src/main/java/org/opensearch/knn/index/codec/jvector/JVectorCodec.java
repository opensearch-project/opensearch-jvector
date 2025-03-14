/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;

public class JVectorCodec extends FilterCodec {

    public static final String CODEC_NAME = "JVectorCodec";
    private int minBatchSizeForQuantization;

    public JVectorCodec() {
        this(CODEC_NAME, new Lucene101Codec(), JVectorFormat.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION);
    }

    public JVectorCodec(int minBatchSizeForQuantization) {
        this(CODEC_NAME, new Lucene101Codec(), minBatchSizeForQuantization);
    }

    public JVectorCodec(String codecName, Codec delegate, int minBatchSizeForQuantization) {
        super(codecName, delegate);
        this.minBatchSizeForQuantization = minBatchSizeForQuantization;
    }

    @Override
    public KnnVectorsFormat knnVectorsFormat() {
        return new JVectorFormat(minBatchSizeForQuantization);
    }

    @Override
    public CompoundFormat compoundFormat() {
        return new JVectorCompoundFormat(delegate.compoundFormat());
    }
}

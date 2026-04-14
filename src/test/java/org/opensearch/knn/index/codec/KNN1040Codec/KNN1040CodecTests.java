/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;
import org.opensearch.knn.index.codec.CustomCodec;
import org.opensearch.knn.index.codec.CustomCodecNoStoredFields;
import org.opensearch.knn.index.codec.KNNCodecTestCase;
import org.opensearch.knn.index.codec.KNNCodecVersion;
import org.opensearch.knn.index.codec.jvector.JVectorFormat;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;

import lombok.SneakyThrows;

@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
public class KNN1040CodecTests extends KNNCodecTestCase {
    @SneakyThrows
    public void testMultiFieldsKnnIndex() {
        testMultiFieldsKnnIndex(KNN1040Codec.builder().knnVectorsFormat(new PerFieldKnnVectorsFormat() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return new JVectorFormat();
            }
        }).delegate(KNNCodecVersion.CURRENT_DEFAULT_DELEGATE).build());
    }

    @SneakyThrows
    public void testMultiFieldsKnnIndexCustomCodecWithStoredFields() {
        testMultiFieldsKnnIndex(KNN1040Codec.builder().knnVectorsFormat(new PerFieldKnnVectorsFormat() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return new JVectorFormat();
            }
        }).delegate(new CustomCodec()).build());
    }

    @SneakyThrows
    public void testMultiFieldsKnnIndexCustomCodecWithoutStoredFields() {
        testMultiFieldsKnnIndex(KNN1040Codec.builder().knnVectorsFormat(new PerFieldKnnVectorsFormat() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return new JVectorFormat();
            }
        }).delegate(new CustomCodecNoStoredFields()).build());
    }
}

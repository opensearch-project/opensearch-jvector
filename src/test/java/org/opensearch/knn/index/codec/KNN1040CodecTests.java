/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import static org.junit.Assert.assertTrue;

import org.apache.lucene.codecs.Codec;
import org.opensearch.knn.index.codec.KNN10010Codec.KNN10010DerivedSourceStoredFieldsFormat;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundFormat;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80DocValuesFormat;
import org.opensearch.knn.index.codec.KNN9120Codec.KNN9120PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.engine.CommonTestUtils;

public class KNN1040CodecTests {

    // Ensure that the codec is able to return the correct doc values format for codec
    public void testCodecSetsDocValuesFormat() {
        final Codec codec = CommonTestUtils.getCodec();
        assertTrue(codec.docValuesFormat() instanceof KNN80DocValuesFormat);
    }

    // Ensure that the codec is able to return the correct compound format for codec
    public void testCodecSetsCompoundFormat() {
        final Codec codec = CommonTestUtils.getCodec();
        assertTrue(codec.compoundFormat() instanceof KNN80CompoundFormat);
    }

    // Ensure that the codec is able to return the correct per field knn vectors format for codec
    public void testCodecSetsCustomPerFieldKnnVectorsFormat() {
        final Codec codec = CommonTestUtils.getCodec();
        assertTrue(codec.knnVectorsFormat() instanceof KNN9120PerFieldKnnVectorsFormat);
    }

    // Ensure that the codec is able to return the correct stored fields format for codec
    public void testCodecSetsStoredFieldsFormat() {
        final Codec codec = CommonTestUtils.getCodec();
        assertTrue(codec.storedFieldsFormat() instanceof KNN10010DerivedSourceStoredFieldsFormat);
    }
}

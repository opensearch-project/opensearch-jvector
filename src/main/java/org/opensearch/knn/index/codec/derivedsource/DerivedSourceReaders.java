/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.util.IOUtils;
import org.opensearch.common.Nullable;

import java.io.Closeable;
import java.io.IOException;

/**
 * Class holds the readers necessary to implement derived source. Important to note that if a segment does not have
 * any of these fields, the values will be null. Caller needs to check if these are null before using.
 */
@RequiredArgsConstructor
@Getter
public class DerivedSourceReaders implements Closeable {
    @Nullable
    private final KnnVectorsReader knnVectorsReader;
    @Nullable
    private final DocValuesProducer docValuesProducer;

    /**
     * Returns this DerivedSourceReaders object with incremented reference count
     *
     * @return DerivedSourceReaders object with incremented reference count
     */
    public DerivedSourceReaders cloneWithMerge() {
        // For cloning, we don't need to reference count. In Lucene, the merging will actually not close any of the
        // readers, so it should only be handled by the original code. See
        // https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/index/IndexWriter.java#L3372
        // for more details
        return this;
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(knnVectorsReader, docValuesProducer);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.StoredFieldVisitor;
import org.opensearch.common.CheckedFunction;
import org.opensearch.core.common.bytes.BytesReference;

import java.io.IOException;

public class KNN9120DerivedSourceStoredFieldsReader extends StoredFieldsReader {

    static final String DERIVED_VECTOR_FIELD_ATTRIBUTE_KEY = "knn-derived-source-enabled";
    static final String DERIVED_VECTOR_FIELD_ATTRIBUTE_TRUE_VALUE = "true";

    private final StoredFieldsReader delegate;

    KNN9120DerivedSourceStoredFieldsReader(StoredFieldsReader in, CheckedFunction<Integer, BytesReference, IOException> sourceProvider) {
        this.delegate = in;
    }

    @Override
    public void checkIntegrity() throws IOException {
        delegate.checkIntegrity();
    }

    @Override
    public void close() throws IOException {
        delegate.close();
    }

    @Override
    public StoredFieldsReader clone() {
        return delegate.clone();
    }

    @Override
    public void document(int docId, StoredFieldVisitor visitor) throws IOException {
        delegate.document(docId, visitor);
    }

    /**
     * Checks if any of the segments being merged contains legacy segments. If so,
     * we need to use the legacy codec
     * for merging.
     *
     * @param mergeState {@link MergeState}
     * @return true if any of the segments being merged contains legacy segments,
     *         false otherwise
     */
    public static boolean doesMergeContainLegacySegments(MergeState mergeState) {
        for (int i = 0; i < mergeState.storedFieldsReaders.length; i++) {
            if (mergeState.storedFieldsReaders[i] instanceof KNN9120DerivedSourceStoredFieldsReader
                && doesSegmentContainLegacyFields(mergeState.fieldInfos[i])) {
                return true;
            }
        }
        return false;
    }

    private static boolean doesSegmentContainLegacyFields(FieldInfos fieldInfos) {
        for (FieldInfo fieldInfo : fieldInfos) {
            if (DERIVED_VECTOR_FIELD_ATTRIBUTE_TRUE_VALUE.equals(fieldInfo.attributes().get(DERIVED_VECTOR_FIELD_ATTRIBUTE_KEY))) {
                return true;
            }
        }
        return false;
    }
}

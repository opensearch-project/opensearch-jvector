/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.CheckedSupplier;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil;

import java.io.IOException;

public abstract class AbstractPerFieldDerivedVectorTransformer implements PerFieldDerivedVectorTransformer {
    /**
     * Utility method for formatting the vector values based on the vector data type.
     *
     * @param fieldInfo fieldinfo for the vector field
     * @param vectorSupplier supplies vector (without clone)
     * @param vectorCloneSupplier supplies clone of vector.
     * @return vector formatted based on the vector data type. Typically, this will be a float[] or int[].
     * @throws IOException if unable to deserialize stored vector
     */
    protected Object formatVector(
        FieldInfo fieldInfo,
        CheckedSupplier<Object, IOException> vectorSupplier,
        CheckedSupplier<Object, IOException> vectorCloneSupplier
    ) throws IOException {
        Object vectorValue = vectorSupplier.get();
        // If the vector value is a byte[], we must deserialize
        if (vectorValue instanceof byte[]) {
            BytesRef vectorBytesRef = new BytesRef((byte[]) vectorValue);
            // Determine the vector data type based on the vector encoding to ensure proper deserialization
            VectorDataType vectorDataType;
            if (fieldInfo.hasVectorValues()) {
                // For vectors with native Lucene encoding, use the encoding type
                if (fieldInfo.getVectorEncoding() == VectorEncoding.BYTE) {
                    vectorDataType = VectorDataType.BYTE;
                } else {
                    vectorDataType = VectorDataType.FLOAT;
                }
            } else {
                // For binary doc values, use the data type from field attributes
                vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
            }
            return KNNVectorFieldMapperUtil.deserializeStoredVector(vectorBytesRef, vectorDataType);
        }
        return vectorCloneSupplier.get();
    }
}

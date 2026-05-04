/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;

public class NestedPerFieldDerivedVectorTransformer extends AbstractPerFieldDerivedVectorTransformer {

    private final FieldInfo childFieldInfo;
    private final DerivedSourceReaders derivedSourceReaders;
    private KNNVectorValues<?> vectorValues;
    private int parentDocId = -1;
    private int currentVectorDocId = -1;

    /**
     *
     * @param childFieldInfo FieldInfo of the child field
     * @param derivedSourceReaders Readers for access segment info
     */
    public NestedPerFieldDerivedVectorTransformer(FieldInfo childFieldInfo, DerivedSourceReaders derivedSourceReaders) {
        this.childFieldInfo = childFieldInfo;
        this.derivedSourceReaders = derivedSourceReaders;
    }

    @Override
    public Object apply(Object object) {
        // This control is needed due to masking operation
        if ((object instanceof Byte && ((Byte) object)
                .byteValue() == org.opensearch.knn.index.codec.KNN10010Codec.KNN10010DerivedSourceStoredFieldsWriter.MASK)
                || (object instanceof Integer && ((Integer) object)
                        .intValue() == org.opensearch.knn.index.codec.KNN10010Codec.KNN10010DerivedSourceStoredFieldsWriter.MASK)) {
            object = null;
        }

        if (object != null) {
            return object;
        }

        // When object is null or MASK, it means the field is not in the source, so we need to inject the vector
        try {
            // Iterate through children until we find one with a vector for this field
            while (currentVectorDocId != DocIdSetIterator.NO_MORE_DOCS && currentVectorDocId < parentDocId) {
                Object vector = formatVector(childFieldInfo, vectorValues::getVector, vectorValues::conditionalCloneVector);

                // If vector is null, this child document doesn't have this field - skip to next
                if (vector == null) {
                    currentVectorDocId = vectorValues.nextDoc();
                    continue;
                }

                currentVectorDocId = vectorValues.nextDoc();
                return vector;
            }
            return null;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void setCurrentDoc(int offset, int docId) throws IOException {
        this.vectorValues = KNNVectorValuesFactory.getVectorValues(
            childFieldInfo,
            derivedSourceReaders.getDocValuesProducer(),
            derivedSourceReaders.getKnnVectorsReader()
        );
        this.parentDocId = docId;
        this.currentVectorDocId = vectorValues.advance(offset);
    }
}

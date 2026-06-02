/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.io.IOException;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;

public class JVectorVectorScorer implements VectorScorer {
    private final JVectorFloatVectorValues floatVectorValues;
    private final KnnVectorValues.DocIndexIterator docIndexIterator;
    private final VectorFloat<?> target;
    private final VectorSimilarityFunction similarityFunction;
    private final org.apache.lucene.index.VectorSimilarityFunction luceneSimilarityFunction;

    public JVectorVectorScorer(
        JVectorFloatVectorValues vectorValues,
        VectorFloat<?> target,
        VectorSimilarityFunction similarityFunction,
        org.apache.lucene.index.VectorSimilarityFunction luceneSimilarityFunction
    ) {
        this.floatVectorValues = vectorValues;
        this.docIndexIterator = floatVectorValues.iterator();
        this.target = target;
        this.similarityFunction = similarityFunction;
        this.luceneSimilarityFunction = luceneSimilarityFunction;
    }

    @Override
    public float score() throws IOException {
        int index = docIndexIterator.index();
        if (index == GraphNodeIdToDocMap.NO_VECTOR_OR_DELETED_DOC) {
            return 0.0f;
        }
        float score = similarityFunction.compare(target, floatVectorValues.vectorFloatValue(index));

        // Apply score transformation for MAXIMUM_INNER_PRODUCT
        // Lucene's MAXIMUM_INNER_PRODUCT formula is: 1 + dotProduct
        // jVector's DOT_PRODUCT returns: (1 + dotProduct) / 2
        // To convert: score * 2 = (1 + dotProduct) / 2 * 2 = 1 + dotProduct
        if (luceneSimilarityFunction == org.apache.lucene.index.VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
            && similarityFunction == VectorSimilarityFunction.DOT_PRODUCT) {
            return score * 2.0f;
        }

        return score;
    }

    @Override
    public DocIdSetIterator iterator() {
        return docIndexIterator;
    }
}

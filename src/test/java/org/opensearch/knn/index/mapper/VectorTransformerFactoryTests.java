/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.mapper;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

public class VectorTransformerFactoryTests extends KNNTestCase {
    public void testAllSpaceTypes_withLucene() {
        for (SpaceType spaceType : SpaceType.values()) {
            VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(KNNEngine.LUCENE, spaceType);
            validateTransformer(spaceType, KNNEngine.LUCENE, transformer);
        }
    }

    public void testAllSpaceTypes_withJVector() {
        for (SpaceType spaceType : SpaceType.values()) {
            VectorTransformer transformer = VectorTransformerFactory.getVectorTransformer(KNNEngine.JVECTOR, spaceType);
            validateTransformer(spaceType, KNNEngine.JVECTOR, transformer);
        }
    }

    private static void validateTransformer(SpaceType spaceType, KNNEngine engine, VectorTransformer transformer) {
        assertSame(
            "Should return NOOP transformer for " + engine + " with " + spaceType,
            VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER,
            transformer
        );
    }
}

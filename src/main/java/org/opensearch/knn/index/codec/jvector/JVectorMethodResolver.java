/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import java.util.HashMap;

import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.AbstractMethodResolver;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;

public class JVectorMethodResolver extends AbstractMethodResolver {

    @Override
    public ResolvedMethodContext resolveMethod(
        KNNMethodContext knnMethodContext,
        KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        SpaceType spaceType
    ) {
        return ResolvedMethodContext.builder()
            .knnMethodContext(initResolvedKNNMethodContext(knnMethodContext, KNNEngine.JVECTOR, spaceType, KNNConstants.DISK_ANN))
            .compressionLevel(CompressionLevel.x1)
            .build();
    }

    protected KNNMethodContext initResolvedKNNMethodContext(
        KNNMethodContext originalMethodContext,
        KNNEngine knnEngine,
        SpaceType spaceType,
        String methodName
    ) {
        if (originalMethodContext == null) {
            return new KNNMethodContext(knnEngine, spaceType, new MethodComponentContext(methodName, new HashMap<>()));
        } else {
            return new KNNMethodContext(originalMethodContext);
        }
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.engine.model.QueryContext;
import org.opensearch.knn.index.query.request.MethodParameter;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.*;

/**
 * Default HNSW context for all engines. Have a different implementation if engine context differs.
 */
public final class JVectorDiskANNSearchContext implements KNNLibrarySearchContext {

    private final Map<String, Parameter<?>> supportedMethodParameters = ImmutableMap.<String, Parameter<?>>builder()
        .put(
            MethodParameter.EF_SEARCH.getName(),
            new Parameter.IntegerParameter(MethodParameter.EF_SEARCH.getName(), null, (value, context) -> true)
        )
        .put(
            MethodParameter.OVERQUERY_FACTOR.getName(),
            new Parameter.IntegerParameter(MethodParameter.OVERQUERY_FACTOR.getName(), DEFAULT_OVER_QUERY_FACTOR, (value, context) -> true)
        )
        .put(MethodParameter.THRESHOLD.getName(),
                new Parameter.DoubleParameter(MethodParameter.THRESHOLD.getName(), DEFAULT_QUERY_SIMILARITY_THRESHOLD, (value, context) -> true)
        )
        .put(MethodParameter.RERANK_FLOOR.getName(),
                new Parameter.DoubleParameter(MethodParameter.RERANK_FLOOR.getName(), DEFAULT_QUERY_RERANK_FLOOR, (value, context) -> true)
        )
        .build();

    @Override
    public Map<String, Parameter<?>> supportedMethodParameters(QueryContext ctx) {
        return supportedMethodParameters;
    }
}

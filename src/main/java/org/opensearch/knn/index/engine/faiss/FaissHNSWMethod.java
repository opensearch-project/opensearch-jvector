/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.AbstractKNNMethod;
import org.opensearch.knn.index.engine.DefaultHnswSearchContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.Parameter;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_ENCODER_TYPE;
import static org.opensearch.knn.common.KNNConstants.FAISS_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.FAISS_PQ_SUBSPACES;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;

/**
 * Faiss HNSW method implementation for float vectors.
 * Supports optional quantization via the {@code faiss.encoder_type} parameter:
 * <ul>
 *   <li>{@code "sq"} — Scalar Quantization (8-bit), appends {@code ,SQ8} to the Faiss factory string</li>
 *   <li>{@code "pq"} — Product Quantization, appends {@code ,PQ<n>} to the Faiss factory string;
 *       requires {@code faiss.pq_subspaces} to be set and to evenly divide the field dimension</li>
 * </ul>
 */
public class FaissHNSWMethod extends AbstractKNNMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(
        VectorDataType.FLOAT
    );

    public static final List<SpaceType> SUPPORTED_SPACES = Arrays.asList(
        SpaceType.UNDEFINED,
        SpaceType.L2,
        SpaceType.COSINESIMIL,
        SpaceType.INNER_PRODUCT
    );

    public static final MethodComponent HNSW_METHOD_COMPONENT = initMethodComponent();

    public FaissHNSWMethod() {
        super(HNSW_METHOD_COMPONENT, Set.copyOf(SUPPORTED_SPACES), new DefaultHnswSearchContext());
    }

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(METHOD_HNSW)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(
                METHOD_PARAMETER_M,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_M,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M,
                    (v, context) -> v > 0
                )
            )
            .addParameter(
                METHOD_PARAMETER_EF_CONSTRUCTION,
                new Parameter.IntegerParameter(
                    METHOD_PARAMETER_EF_CONSTRUCTION,
                    KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION,
                    (v, context) -> v > 0
                )
            )
            .addParameter(
                FAISS_ENCODER_TYPE,
                new Parameter.StringParameter(
                    FAISS_ENCODER_TYPE,
                    null,
                    (v, context) -> v == null || ENCODER_SQ.equals(v) || ENCODER_PQ.equals(v)
                )
            )
            .addParameter(
                FAISS_PQ_SUBSPACES,
                new Parameter.IntegerParameter(
                    FAISS_PQ_SUBSPACES,
                    null,
                    (v, context) -> v == null || (v > 0 && context.getDimension() % v == 0)
                )
            )
            .addParameter(
                FAISS_EF_SEARCH,
                new Parameter.IntegerParameter(
                    FAISS_EF_SEARCH,
                    null,
                    (v, context) -> v == null || v > 0
                )
            )
            .build();
    }
}

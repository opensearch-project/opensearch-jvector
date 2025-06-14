/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import com.google.common.collect.ImmutableSet;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.*;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.*;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;

public class JVectorDiskANNMethod extends AbstractKNNMethod {

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    public final static List<SpaceType> SUPPORTED_SPACES = Arrays.asList(
        SpaceType.UNDEFINED,
        SpaceType.L2,
        SpaceType.L1,
        SpaceType.LINF,
        SpaceType.COSINESIMIL,
        SpaceType.INNER_PRODUCT
    );

    final static MethodComponent DISK_ANN_METHOD_COMPONENT = initMethodComponent();

    private static MethodComponent initMethodComponent() {
        return MethodComponent.Builder.builder(DISK_ANN)
            .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
            .addParameter(
                METHOD_PARAMETER_M,
                new Parameter.IntegerParameter(METHOD_PARAMETER_M, KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_M, (v, context) -> v > 0)
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
                METHOD_PARAMETER_ALPHA,
                new Parameter.DoubleParameter(METHOD_PARAMETER_ALPHA, KNNConstants.DEFAULT_ALPHA_VALUE, (v, context) -> v > 0)
            )
            .addParameter(
                METHOD_PARAMETER_NEIGHBOR_OVERFLOW,
                new Parameter.DoubleParameter(
                    METHOD_PARAMETER_NEIGHBOR_OVERFLOW,
                    KNNConstants.DEFAULT_NEIGHBOR_OVERFLOW_VALUE,
                    (v, context) -> v > 0
                )
            )
            .addParameter(
                METHOD_PARAMETER_HIERARCHY_ENABLED,
                new Parameter.BooleanParameter(
                    METHOD_PARAMETER_HIERARCHY_ENABLED,
                    KNNConstants.DEFAULT_HIERARCHY_ENABLED,
                    (v, context) -> true
                )
            )
            .build();
    }

    public JVectorDiskANNMethod() {
        super(DISK_ANN_METHOD_COMPONENT, Set.copyOf(SUPPORTED_SPACES), new JVectorDiskANNSearchContext());
    }
}

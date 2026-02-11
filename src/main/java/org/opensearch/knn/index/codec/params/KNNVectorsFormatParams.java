/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.params;

import lombok.Getter;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.jvector.JVectorFormat;

import java.util.Map;
import java.util.function.Function;

/**
 * Class provides params for LuceneHNSWVectorsFormat
 */
@Getter
public class KNNVectorsFormatParams {
    private int maxConnections;
    private int beamWidth;
    private float alpha;
    private float neighborOverflow;
    private int minBatchSizeForQuantization;
    private boolean hierarchyEnabled;
    private Function<Integer, Integer> numberOfSubspacesPerVectorSupplier;
    private final SpaceType spaceType;
    private boolean leadingSegmentMergeDisabled;

    public KNNVectorsFormatParams(final Map<String, Object> params, int defaultMaxConnections, int defaultBeamWidth) {
        this(
            params,
            defaultMaxConnections,
            defaultBeamWidth,
            KNNConstants.DEFAULT_ALPHA_VALUE.floatValue(),
            KNNConstants.DEFAULT_NEIGHBOR_OVERFLOW_VALUE.floatValue(),
            KNNConstants.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION,
            KNNConstants.DEFAULT_HIERARCHY_ENABLED,
            SpaceType.UNDEFINED
        );
    }

    public KNNVectorsFormatParams(
        final Map<String, Object> params,
        int defaultMaxConnections,
        int defaultBeamWidth,
        float defaultAlpha,
        float defaultNeighborOverflow,
        int defaultMinBatchSizeForQuantization,
        boolean defaultHierarchyEnabled,
        SpaceType spaceType
    ) {
        initMaxConnections(params, defaultMaxConnections);
        initBeamWidth(params, defaultBeamWidth);
        initAlpha(params, defaultAlpha);
        initNeighborOverflow(params, defaultNeighborOverflow);
        initMinBatchSizeForQuantization(params, defaultMinBatchSizeForQuantization);
        initHierarchyEnabled(params, defaultHierarchyEnabled);
        initNumberOfSubspacesPerVectorSupplier(params);
        this.spaceType = spaceType;
        initLeadingSegmentMergeDisabled(params, KNNConstants.DEFAULT_LEADING_SEGMENT_MERGE_DISABLED);
    }

    public boolean validate(final Map<String, Object> params) {
        return true;
    }

    private void initMaxConnections(final Map<String, Object> params, int defaultMaxConnections) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_M)) {
            this.maxConnections = (int) params.get(KNNConstants.METHOD_PARAMETER_M);
            return;
        }
        this.maxConnections = defaultMaxConnections;
    }

    private void initBeamWidth(final Map<String, Object> params, int defaultBeamWidth) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION)) {
            this.beamWidth = (int) params.get(KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION);
            return;
        }
        this.beamWidth = defaultBeamWidth;
    }

    private void initAlpha(final Map<String, Object> params, float defaultAlpha) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_ALPHA)) {
            this.alpha = ((Double) params.get(KNNConstants.METHOD_PARAMETER_ALPHA)).floatValue();
            return;
        }
        this.alpha = defaultAlpha;
    }

    private void initNeighborOverflow(final Map<String, Object> params, float defaultNeighborOverflow) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_NEIGHBOR_OVERFLOW)) {
            this.neighborOverflow = ((Double) params.get(KNNConstants.METHOD_PARAMETER_NEIGHBOR_OVERFLOW)).floatValue();
            return;
        }
        this.neighborOverflow = defaultNeighborOverflow;
    }

    private void initMinBatchSizeForQuantization(final Map<String, Object> params, int defaultMinBatchSizeForQuantization) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_MIN_BATCH_SIZE_FOR_QUANTIZATION)) {
            this.minBatchSizeForQuantization = (int) params.get(KNNConstants.METHOD_PARAMETER_MIN_BATCH_SIZE_FOR_QUANTIZATION);
            return;
        }
        this.minBatchSizeForQuantization = defaultMinBatchSizeForQuantization;
    }

    private void initHierarchyEnabled(final Map<String, Object> params, boolean defaultHierarchyEnabled) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_HIERARCHY_ENABLED)) {
            this.hierarchyEnabled = (boolean) params.get(KNNConstants.METHOD_PARAMETER_HIERARCHY_ENABLED);
            return;
        }
        this.hierarchyEnabled = defaultHierarchyEnabled;
    }

    private void initNumberOfSubspacesPerVectorSupplier(final Map<String, Object> params) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_NUM_PQ_SUBSPACES)) {
            int numPQSubspaces = (int) params.get(KNNConstants.METHOD_PARAMETER_NUM_PQ_SUBSPACES);
            this.numberOfSubspacesPerVectorSupplier = (originalDimension) -> numPQSubspaces;
            return;
        }
        this.numberOfSubspacesPerVectorSupplier = JVectorFormat::getDefaultNumberOfSubspacesPerVector;
    }

    private void initLeadingSegmentMergeDisabled(final Map<String, Object> params, boolean defaultLsmDisabled) {
        if (params != null && params.containsKey(KNNConstants.METHOD_PARAMETER_LEADING_SEGMENT_MERGE_DISABLED)) {
            this.hierarchyEnabled = (boolean) params.get(KNNConstants.METHOD_PARAMETER_LEADING_SEGMENT_MERGE_DISABLED);
            return;
        }
        this.leadingSegmentMergeDisabled = defaultLsmDisabled;
    }
}

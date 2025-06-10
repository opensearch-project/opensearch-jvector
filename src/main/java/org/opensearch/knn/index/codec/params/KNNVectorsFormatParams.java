/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.params;

import lombok.Getter;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;

import java.util.Map;

/**
 * Class provides params for LuceneHNSWVectorsFormat
 */
@Getter
public class KNNVectorsFormatParams {
    private int maxConnections;
    private int beamWidth;
    private float alpha;
    private float neighborOverflow;
    private final SpaceType spaceType;

    public KNNVectorsFormatParams(final Map<String, Object> params, int defaultMaxConnections, int defaultBeamWidth) {
        this(
            params,
            defaultMaxConnections,
            defaultBeamWidth,
            KNNConstants.DEFAULT_ALPHA_VALUE.floatValue(),
            KNNConstants.DEFAULT_NEIGHBOR_OVERFLOW_VALUE.floatValue(),
            SpaceType.UNDEFINED
        );
    }

    public KNNVectorsFormatParams(
        final Map<String, Object> params,
        int defaultMaxConnections,
        int defaultBeamWidth,
        float defaultAlpha,
        float defaultNeighborOverflow,
        SpaceType spaceType
    ) {
        initMaxConnections(params, defaultMaxConnections);
        initBeamWidth(params, defaultBeamWidth);
        initAlpha(params, defaultAlpha);
        initNeighborOverflow(params, defaultNeighborOverflow);
        this.spaceType = spaceType;
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
}

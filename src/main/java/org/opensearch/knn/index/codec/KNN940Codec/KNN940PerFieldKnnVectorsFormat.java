/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN940Codec;

import org.apache.lucene.backward_codecs.lucene94.Lucene94HnswVectorsFormat;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.BasePerFieldKnnVectorsFormat;

import java.util.Optional;

/**
 * Class provides per field format implementation for Lucene Knn vector type
 */
public class KNN940PerFieldKnnVectorsFormat extends BasePerFieldKnnVectorsFormat {

    public KNN940PerFieldKnnVectorsFormat(final Optional<MapperService> mapperService) {
        super(
            mapperService,
            Lucene94HnswVectorsFormat.DEFAULT_MAX_CONN,
            Lucene94HnswVectorsFormat.DEFAULT_BEAM_WIDTH,
            KNNConstants.DEFAULT_ALPHA_VALUE.floatValue(),
            KNNConstants.DEFAULT_NEIGHBOR_OVERFLOW_VALUE.floatValue(),
            () -> new Lucene94HnswVectorsFormat(),
            (knnEngine, knnVectorsFormatParams) -> new Lucene94HnswVectorsFormat(
                knnVectorsFormatParams.getMaxConnections(),
                knnVectorsFormatParams.getBeamWidth()
            )
        );
    }
}

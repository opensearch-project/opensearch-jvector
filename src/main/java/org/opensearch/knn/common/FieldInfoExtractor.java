/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import lombok.experimental.UtilityClass;
import org.apache.commons.lang.StringUtils;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.engine.qframe.QuantizationConfigParser;

import static org.opensearch.knn.common.KNNConstants.QFRAMEWORK_CONFIG;

import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

/**
 * A utility class to extract information from FieldInfo and also provides utility functions to extract fieldInfo
 */
@UtilityClass
public class FieldInfoExtractor {

    /**
     * Extracts KNNEngine from FieldInfo
     * @param field {@link FieldInfo}
     * @return {@link KNNEngine}
     */
    public static KNNEngine extractKNNEngine(final FieldInfo field) {
        final String engineName = field.attributes().getOrDefault(KNNConstants.KNN_ENGINE, KNNEngine.DEFAULT.getName());
        return KNNEngine.getEngine(engineName);
    }

    /**
     * Extracts VectorDataType from FieldInfo
     * @param fieldInfo {@link FieldInfo}
     * @return {@link VectorDataType}
     */
    public static VectorDataType extractVectorDataType(final FieldInfo fieldInfo) {
        String vectorDataTypeString = fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD);
        return StringUtils.isNotEmpty(vectorDataTypeString) ? VectorDataType.get(vectorDataTypeString) : VectorDataType.DEFAULT;
    }

    /**
     * Extract quantization config from fieldInfo
     *
     * @param fieldInfo {@link FieldInfo}
     * @return {@link QuantizationConfig}
     */
    public static QuantizationConfig extractQuantizationConfig(final FieldInfo fieldInfo) {
        String quantizationConfigString = fieldInfo.getAttribute(QFRAMEWORK_CONFIG);
        if (StringUtils.isEmpty(quantizationConfigString)) {
            return QuantizationConfig.EMPTY;
        }
        return QuantizationConfigParser.fromCsv(quantizationConfigString);
    }

    /**
     * Get the space type for the given field info.
     *
     * @param fieldInfo FieldInfo instance to extract space type from
     * @return SpaceType for the given field info
     */
    public static SpaceType getSpaceType(final FieldInfo fieldInfo) {
        final String spaceTypeString = fieldInfo.getAttribute(SPACE_TYPE);
        if (StringUtils.isNotEmpty(spaceTypeString)) {
            return SpaceType.getSpace(spaceTypeString);
        }

        throw new IllegalStateException("Space type is not set for field: " + fieldInfo.name);
    }

    /**
     * Get the field info for the given field name, do a null check on the fieldInfo, as this function can return null,
     * if the field is not found.
     * @param leafReader {@link LeafReader}
     * @param fieldName {@link String}
     * @return {@link FieldInfo}
     */
    public static @Nullable FieldInfo getFieldInfo(final LeafReader leafReader, final String fieldName) {
        return leafReader.getFieldInfos().fieldInfo(fieldName);
    }
}

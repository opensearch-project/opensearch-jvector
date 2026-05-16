/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.opensearch.Version;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.mapper.DocumentMapper;
import org.opensearch.index.mapper.FieldMapper;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.mapper.MappingLookup;
import org.opensearch.index.mapper.SourceFieldMapper;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

public class IndexUtilTests extends KNNTestCase {

    public void testUpdateVectorDataTypeToParameters_whenVectorDataTypeIsBinary() {
        Map<String, Object> indexParams = new HashMap<>();
        IndexUtil.updateVectorDataTypeToParameters(indexParams, VectorDataType.BINARY);
        assertEquals(VectorDataType.BINARY.getValue(), indexParams.get(VECTOR_DATA_TYPE_FIELD));
    }

    public void testIsDerivedEnabledForField_withVariousIncludeExcludeCombinations() {
        // Test 1: No includes, no excludes → field not excluded → derived enabled
        assertDerivedEnabledWithSourceFiltering("my_vector", Collections.emptyList(), Collections.emptyList(), true);

        // Test 2: Only includes specified, field matches → field not excluded → derived enabled
        assertDerivedEnabledWithSourceFiltering("my_vector", List.of("my_vector", "title"), Collections.emptyList(), true);

        // Test 3: Only includes specified, field doesn't match → field excluded → derived disabled
        assertDerivedEnabledWithSourceFiltering("my_vector", List.of("title", "author"), Collections.emptyList(), false);

        // Test 4: Only excludes specified, field matches → field excluded → derived disabled
        assertDerivedEnabledWithSourceFiltering("my_vector", Collections.emptyList(), List.of("my_vector"), false);

        // Test 5: Only excludes specified, field doesn't match → field not excluded → derived enabled
        assertDerivedEnabledWithSourceFiltering("my_vector", Collections.emptyList(), List.of("other_field"), true);

        // Test 6: Both specified, field matches include but also matches exclude → field excluded → derived disabled
        assertDerivedEnabledWithSourceFiltering("my_vector", List.of("my_vector", "title"), List.of("my_vector"), false);

        // Test 7: Both specified, field matches include and doesn't match exclude → field not excluded → derived enabled
        assertDerivedEnabledWithSourceFiltering("my_vector", List.of("my_vector", "title"), List.of("other_field"), true);

        // Test 8: Wildcard includes, field matches → field not excluded → derived enabled
        assertDerivedEnabledWithSourceFiltering("my_vector", List.of("my_*"), Collections.emptyList(), true);

        // Test 9: Wildcard includes, field doesn't match → field excluded → derived disabled
        assertDerivedEnabledWithSourceFiltering("my_vector", List.of("other_*"), Collections.emptyList(), false);

        // Test 10: Wildcard excludes, field matches → field excluded → derived disabled
        assertDerivedEnabledWithSourceFiltering("my_vector", Collections.emptyList(), List.of("my_*"), false);

        // Test 11: Wildcard excludes, field doesn't match → field not excluded → derived enabled
        assertDerivedEnabledWithSourceFiltering("my_vector", Collections.emptyList(), List.of("other_*"), true);

        // Test 12: Nested field with wildcard includes
        assertDerivedEnabledWithSourceFiltering("user.name.embedding", List.of("user.*"), Collections.emptyList(), true);

        // Test 13: Nested field not matching wildcard includes
        assertDerivedEnabledWithSourceFiltering("product.embedding", List.of("user.*"), Collections.emptyList(), false);

        // Test 14: Null sourceMapper → field not excluded → derived enabled
        assertDerivedEnabledWithNullSourceMapper("my_vector", true);
    }

    private void assertDerivedEnabledWithSourceFiltering(
        String fieldName,
        Collection<String> includes,
        Collection<String> excludes,
        boolean expectedDerivedEnabled
    ) {
        KNNVectorFieldType knnVectorFieldType = mock(KNNVectorFieldType.class);
        when(knnVectorFieldType.name()).thenReturn(fieldName);

        MapperService mapperService = mock(MapperService.class);
        DocumentMapper documentMapper = mock(DocumentMapper.class);
        SourceFieldMapper sourceFieldMapper = mock(SourceFieldMapper.class);
        MappingLookup mappingLookup = mock(MappingLookup.class);
        FieldMapper fieldMapper = mock(FieldMapper.class);

        when(mapperService.documentMapper()).thenReturn(documentMapper);
        when(documentMapper.metadataMapper(SourceFieldMapper.class)).thenReturn(sourceFieldMapper);
        when(sourceFieldMapper.getIncludes()).thenReturn(includes);
        when(sourceFieldMapper.getExcludes()).thenReturn(excludes);

        // Setup for copyTo check to pass (not affect our test)
        when(documentMapper.mappers()).thenReturn(mappingLookup);
        when(mappingLookup.getMapper(fieldName)).thenReturn(fieldMapper);
        when(fieldMapper.copyTo()).thenReturn(null);

        boolean result = IndexUtil.isDerivedEnabledForField(knnVectorFieldType, mapperService);

        assertEquals(
            String.format(
                Locale.ROOT,
                "Field '%s' with includes=%s, excludes=%s: isDerivedEnabledForField should be %s",
                fieldName,
                includes,
                excludes,
                expectedDerivedEnabled
            ),
            expectedDerivedEnabled,
            result
        );
    }

    private void assertDerivedEnabledWithNullSourceMapper(String fieldName, boolean expectedDerivedEnabled) {
        KNNVectorFieldType knnVectorFieldType = mock(KNNVectorFieldType.class);
        when(knnVectorFieldType.name()).thenReturn(fieldName);

        MapperService mapperService = mock(MapperService.class);
        DocumentMapper documentMapper = mock(DocumentMapper.class);
        MappingLookup mappingLookup = mock(MappingLookup.class);
        FieldMapper fieldMapper = mock(FieldMapper.class);

        when(mapperService.documentMapper()).thenReturn(documentMapper);
        when(documentMapper.metadataMapper(SourceFieldMapper.class)).thenReturn(null);

        // Setup for copyTo check to pass
        when(documentMapper.mappers()).thenReturn(mappingLookup);
        when(mappingLookup.getMapper(fieldName)).thenReturn(fieldMapper);
        when(fieldMapper.copyTo()).thenReturn(null);

        boolean result = IndexUtil.isDerivedEnabledForField(knnVectorFieldType, mapperService);

        assertEquals(expectedDerivedEnabled, result);
    }

    /**
     * Helper to create a MapperService mock with the required call chain for isDerivedEnabledForIndex.
     * Sets up: documentMapper().sourceMapper().enabled(), getIndexSettings().isDerivedSourceEnabled(),
     * getIndexSettings().getSettings(), and getIndexSettings().isSegRepLocalEnabled().
     */
    private MapperService buildMockMapperServiceForDerived(
        boolean sourceEnabled,
        boolean coreDerivedSourceEnabled,
        boolean knnDerivedSourceEnabled,
        boolean segRepLocalEnabled
    ) {
        MapperService mapperService = mock(MapperService.class);

        // Mock documentMapper().sourceMapper().enabled()
        org.opensearch.index.mapper.DocumentMapper documentMapper = mock(org.opensearch.index.mapper.DocumentMapper.class);
        org.opensearch.index.mapper.SourceFieldMapper sourceFieldMapper = mock(org.opensearch.index.mapper.SourceFieldMapper.class);
        when(sourceFieldMapper.enabled()).thenReturn(sourceEnabled);
        when(documentMapper.sourceMapper()).thenReturn(sourceFieldMapper);
        when(mapperService.documentMapper()).thenReturn(documentMapper);

        // Mock getIndexSettings()
        org.opensearch.index.IndexSettings indexSettings = mock(org.opensearch.index.IndexSettings.class);
        when(indexSettings.isDerivedSourceEnabled()).thenReturn(coreDerivedSourceEnabled);
        when(indexSettings.isSegRepLocalEnabled()).thenReturn(segRepLocalEnabled);

        Settings settings = Settings.builder().put(KNNSettings.KNN_DERIVED_SOURCE_ENABLED, knnDerivedSourceEnabled).build();
        when(indexSettings.getSettings()).thenReturn(settings);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);
        when(mapperService.getIndexSettings().getIndexVersionCreated()).thenReturn(Version.CURRENT);

        return mapperService;
    }

    public void testIsDerivedEnabledForIndex_whenCoreDerivedSourceEnabled_thenReturnFalse() {
        // Core derived source is ON → should return false (core takes precedence over KNN derived source)
        MapperService mapperService = buildMockMapperServiceForDerived(true, true, true, false);
        assertFalse(IndexUtil.isDerivedEnabledForIndex(mapperService));
    }

    public void testIsDerivedEnabledForIndex_whenCoreDerivedSourceDisabledAndKnnDerivedSourceEnabled_thenReturnTrue() {
        // Core derived source is OFF, KNN derived source is ON, source enabled, no seg rep → should return true
        MapperService mapperService = buildMockMapperServiceForDerived(true, false, true, false);
        assertTrue(IndexUtil.isDerivedEnabledForIndex(mapperService));
    }

    public void testIsDerivedEnabledForIndex_whenMapperServiceNull_thenReturnFalse() {
        assertFalse(IndexUtil.isDerivedEnabledForIndex(null));
    }

    public void testIsDerivedEnabledForIndex_whenSourceDisabled_thenReturnFalse() {
        MapperService mapperService = buildMockMapperServiceForDerived(false, false, true, false);
        assertFalse(IndexUtil.isDerivedEnabledForIndex(mapperService));
    }

    public void testIsDerivedEnabledForIndex_whenKnnDerivedSourceDisabled_thenReturnFalse() {
        MapperService mapperService = buildMockMapperServiceForDerived(true, false, false, false);
        assertFalse(IndexUtil.isDerivedEnabledForIndex(mapperService));
    }

    public void testIsDerivedEnabledForIndex_whenSegRepLocalEnabled_thenReturnFalse() {
        MapperService mapperService = buildMockMapperServiceForDerived(true, false, true, true);
        assertFalse(IndexUtil.isDerivedEnabledForIndex(mapperService));
    }

}

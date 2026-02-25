/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.index.Index;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.VECTOR_DATA_TYPE_FIELD;

/**
 * Unit tests for MMRVectorFieldInfo
 */
public class MMRVectorFieldInfoTests extends LuceneTestCase {

    // ========== Constructor Tests ==========

    @Test
    public void testDefaultConstructor() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();

        assertNull(info.getIndexName());
        assertNull(info.getFieldPath());
        assertNull(info.getVectorDataType());
        assertNull(info.getSpaceType());
        assertFalse(info.isUnmapped());
        assertNull(info.getFieldType());
    }

    @Test
    public void testParameterizedConstructor() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.FLOAT);

        assertEquals(SpaceType.L2, info.getSpaceType());
        assertEquals(VectorDataType.FLOAT, info.getVectorDataType());
        assertNull(info.getIndexName());
        assertNull(info.getFieldPath());
    }

    @Test
    public void testParameterizedConstructorWithAllSpaceTypes() {
        SpaceType[] spaceTypes = { SpaceType.L2, SpaceType.COSINESIMIL, SpaceType.INNER_PRODUCT };

        for (SpaceType spaceType : spaceTypes) {
            MMRVectorFieldInfo info = new MMRVectorFieldInfo(spaceType, VectorDataType.FLOAT);
            assertEquals(spaceType, info.getSpaceType());
            assertEquals(VectorDataType.FLOAT, info.getVectorDataType());
        }
    }

    @Test
    public void testParameterizedConstructorWithAllVectorDataTypes() {
        VectorDataType[] dataTypes = { VectorDataType.FLOAT, VectorDataType.BYTE, VectorDataType.BINARY };

        for (VectorDataType dataType : dataTypes) {
            MMRVectorFieldInfo info = new MMRVectorFieldInfo(SpaceType.L2, dataType);
            assertEquals(SpaceType.L2, info.getSpaceType());
            assertEquals(dataType, info.getVectorDataType());
        }
    }

    // ========== isKNNVectorField Tests ==========

    @Test
    public void testIsKNNVectorField_WithKNNVectorType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        info.setFieldType(KNNVectorFieldMapper.CONTENT_TYPE);

        assertTrue(info.isKNNVectorField());
    }

    @Test
    public void testIsKNNVectorField_WithNonKNNType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        info.setFieldType("text");

        assertFalse(info.isKNNVectorField());
    }

    @Test
    public void testIsKNNVectorField_WithNullType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        info.setFieldType(null);

        assertFalse(info.isKNNVectorField());
    }

    @Test
    public void testIsKNNVectorField_WithVariousTypes() {
        String[] nonKnnTypes = { "text", "keyword", "long", "integer", "float", "boolean", "date" };

        for (String type : nonKnnTypes) {
            MMRVectorFieldInfo info = new MMRVectorFieldInfo();
            info.setFieldType(type);
            assertFalse("Field type " + type + " should not be KNN vector", info.isKNNVectorField());
        }
    }

    // ========== setKnnConfig Tests ==========

    @Test
    public void testSetKnnConfig_WithTopLevelSpaceType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> config = new HashMap<>();
        config.put(TOP_LEVEL_PARAMETER_SPACE_TYPE, "l2");
        config.put(VECTOR_DATA_TYPE_FIELD, "float");

        info.setKnnConfig(config);

        assertEquals(SpaceType.L2, info.getSpaceType());
        assertEquals(VectorDataType.FLOAT, info.getVectorDataType());
    }

    @Test
    public void testSetKnnConfig_WithMethodLevelSpaceType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> config = new HashMap<>();
        Map<String, Object> method = new HashMap<>();
        method.put(METHOD_PARAMETER_SPACE_TYPE, "cosinesimil");
        config.put(KNN_METHOD, method);
        config.put(VECTOR_DATA_TYPE_FIELD, "byte");

        info.setKnnConfig(config);

        assertEquals(SpaceType.COSINESIMIL, info.getSpaceType());
        assertEquals(VectorDataType.BYTE, info.getVectorDataType());
    }

    @Test
    public void testSetKnnConfig_WithDefaultSpaceType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> config = new HashMap<>();
        config.put(VECTOR_DATA_TYPE_FIELD, "float");
        // No space type specified

        info.setKnnConfig(config);

        // Should use default space type for FLOAT
        assertNotNull(info.getSpaceType());
        assertEquals(VectorDataType.FLOAT, info.getVectorDataType());
    }

    @Test
    public void testSetKnnConfig_WithDefaultVectorDataType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> config = new HashMap<>();
        config.put(TOP_LEVEL_PARAMETER_SPACE_TYPE, "l2");
        // No vector data type specified

        info.setKnnConfig(config);

        assertEquals(SpaceType.L2, info.getSpaceType());
        assertEquals(VectorDataType.DEFAULT, info.getVectorDataType());
    }

    @Test
    public void testSetKnnConfig_TopLevelSpaceTypeTakesPrecedence() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> config = new HashMap<>();
        Map<String, Object> method = new HashMap<>();
        method.put(METHOD_PARAMETER_SPACE_TYPE, "cosinesimil");
        config.put(KNN_METHOD, method);
        config.put(TOP_LEVEL_PARAMETER_SPACE_TYPE, "l2"); // This should take precedence

        info.setKnnConfig(config);

        assertEquals(SpaceType.L2, info.getSpaceType());
    }

    @Test
    public void testSetKnnConfig_WithEmptyConfig() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> config = new HashMap<>();

        info.setKnnConfig(config);

        // Should have default values
        assertNotNull(info.getSpaceType()); // Default space type
        assertEquals(VectorDataType.DEFAULT, info.getVectorDataType());
    }

    @Test
    public void testSetKnnConfig_WithAllSpaceTypes() {
        String[][] spaceTypeTests = {
            { "l2", "l2" },
            { "cosinesimil", "cosinesimil" },
            { "innerproduct", "innerproduct" }
        };

        for (String[] test : spaceTypeTests) {
            MMRVectorFieldInfo info = new MMRVectorFieldInfo();
            Map<String, Object> config = new HashMap<>();
            config.put(TOP_LEVEL_PARAMETER_SPACE_TYPE, test[0]);

            info.setKnnConfig(config);

            assertEquals("Space type mismatch for " + test[0], test[0], info.getSpaceType().getValue());
        }
    }

    @Test
    public void testSetKnnConfig_WithAllVectorDataTypes() {
        String[][] dataTypeTests = {
            { "float", "float" },
            { "byte", "byte" },
            { "binary", "binary" }
        };

        for (String[] test : dataTypeTests) {
            MMRVectorFieldInfo info = new MMRVectorFieldInfo();
            Map<String, Object> config = new HashMap<>();
            config.put(VECTOR_DATA_TYPE_FIELD, test[0]);

            info.setKnnConfig(config);

            assertEquals("Vector data type mismatch for " + test[0], test[0], info.getVectorDataType().getValue());
        }
    }

    @Test
    public void testSetKnnConfig_WithComplexMethodConfig() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> config = new HashMap<>();
        Map<String, Object> method = new HashMap<>();
        method.put("name", "hnsw");
        method.put(METHOD_PARAMETER_SPACE_TYPE, "innerproduct");
        method.put("engine", "jvector");
        config.put(KNN_METHOD, method);
        config.put(VECTOR_DATA_TYPE_FIELD, "float");

        info.setKnnConfig(config);

        assertEquals(SpaceType.INNER_PRODUCT, info.getSpaceType());
        assertEquals(VectorDataType.FLOAT, info.getVectorDataType());
    }

    @Test
    public void testSetKnnConfig_WithMethodButNoSpaceType() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        Map<String, Object> config = new HashMap<>();
        Map<String, Object> method = new HashMap<>();
        method.put("name", "hnsw");
        // No space_type in method
        config.put(KNN_METHOD, method);

        info.setKnnConfig(config);

        // Should use default space type
        assertNotNull(info.getSpaceType());
    }

    // ========== setIndexNameByIndexMetadata Tests ==========

    @Test
    public void testSetIndexNameByIndexMetadata() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();
        
        Index index = new Index("test-index", "test-uuid");
        IndexMetadata indexMetadata = IndexMetadata.builder("test-index")
            .settings(Settings.builder().put(IndexMetadata.SETTING_VERSION_CREATED, org.opensearch.Version.CURRENT))
            .numberOfShards(1)
            .numberOfReplicas(0)
            .build();

        info.setIndexNameByIndexMetadata(indexMetadata);

        assertEquals("test-index", info.getIndexName());
    }

    @Test
    public void testSetIndexNameByIndexMetadata_WithDifferentIndexNames() {
        String[] indexNames = { "index-1", "my-knn-index", "test_index_123", "index.with.dots" };

        for (String indexName : indexNames) {
            MMRVectorFieldInfo info = new MMRVectorFieldInfo();
            IndexMetadata indexMetadata = IndexMetadata.builder(indexName)
                .settings(Settings.builder().put(IndexMetadata.SETTING_VERSION_CREATED, org.opensearch.Version.CURRENT))
                .numberOfShards(1)
                .numberOfReplicas(0)
                .build();

            info.setIndexNameByIndexMetadata(indexMetadata);

            assertEquals(indexName, info.getIndexName());
        }
    }

    // ========== Getter/Setter Tests ==========

    @Test
    public void testSettersAndGetters() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();

        info.setIndexName("test-index");
        assertEquals("test-index", info.getIndexName());

        info.setFieldPath("embedding.vector");
        assertEquals("embedding.vector", info.getFieldPath());

        info.setVectorDataType(VectorDataType.BYTE);
        assertEquals(VectorDataType.BYTE, info.getVectorDataType());

        info.setSpaceType(SpaceType.COSINESIMIL);
        assertEquals(SpaceType.COSINESIMIL, info.getSpaceType());

        info.setUnmapped(true);
        assertTrue(info.isUnmapped());

        info.setFieldType("knn_vector");
        assertEquals("knn_vector", info.getFieldType());
    }

    @Test
    public void testUnmappedFlag() {
        MMRVectorFieldInfo info = new MMRVectorFieldInfo();

        // Default should be false
        assertFalse(info.isUnmapped());

        info.setUnmapped(true);
        assertTrue(info.isUnmapped());

        info.setUnmapped(false);
        assertFalse(info.isUnmapped());
    }

    // ========== Integration Tests ==========

    @Test
    public void testCompleteWorkflow() {
        // Create info with constructor
        MMRVectorFieldInfo info = new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.FLOAT);

        // Set additional fields
        info.setFieldPath("user.embedding");
        info.setFieldType(KNNVectorFieldMapper.CONTENT_TYPE);
        info.setUnmapped(false);

        // Set index metadata
        IndexMetadata indexMetadata = IndexMetadata.builder("my-index")
            .settings(Settings.builder().put(IndexMetadata.SETTING_VERSION_CREATED, org.opensearch.Version.CURRENT))
            .numberOfShards(1)
            .numberOfReplicas(0)
            .build();
        info.setIndexNameByIndexMetadata(indexMetadata);

        // Verify all fields
        assertEquals("my-index", info.getIndexName());
        assertEquals("user.embedding", info.getFieldPath());
        assertEquals(SpaceType.L2, info.getSpaceType());
        assertEquals(VectorDataType.FLOAT, info.getVectorDataType());
        assertTrue(info.isKNNVectorField());
        assertFalse(info.isUnmapped());
    }

    @Test
    public void testConfigOverridesConstructorValues() {
        // Create with initial values
        MMRVectorFieldInfo info = new MMRVectorFieldInfo(SpaceType.L2, VectorDataType.FLOAT);

        assertEquals(SpaceType.L2, info.getSpaceType());
        assertEquals(VectorDataType.FLOAT, info.getVectorDataType());

        // Override with config
        Map<String, Object> config = new HashMap<>();
        config.put(TOP_LEVEL_PARAMETER_SPACE_TYPE, "cosinesimil");
        config.put(VECTOR_DATA_TYPE_FIELD, "byte");

        info.setKnnConfig(config);

        // Values should be overridden
        assertEquals(SpaceType.COSINESIMIL, info.getSpaceType());
        assertEquals(VectorDataType.BYTE, info.getVectorDataType());
    }
}

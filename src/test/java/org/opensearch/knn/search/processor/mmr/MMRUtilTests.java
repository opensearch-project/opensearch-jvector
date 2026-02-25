/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;
import org.opensearch.index.mapper.ObjectMapper;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Unit tests for MMRUtil utility class
 */
public class MMRUtilTests extends LuceneTestCase {

    // ========== Tests for extractVectorFromHit ==========

    @Test
    public void testExtractVectorFromHit_SimpleFloatVector() {
        Map<String, Object> source = Map.of("embedding", List.of(0.1, 0.2, 0.3, 0.4));

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "embedding", "doc-1", true);

        assertNotNull(result);
        assertEquals(4, result.length);
        assertEquals(0.1f, result[0], 0.0001f);
        assertEquals(0.2f, result[1], 0.0001f);
        assertEquals(0.3f, result[2], 0.0001f);
        assertEquals(0.4f, result[3], 0.0001f);
    }

    @Test
    public void testExtractVectorFromHit_SimpleByteVector() {
        Map<String, Object> source = Map.of("embedding", List.of(1.0, 2.0, 3.0, 127.0));

        byte[] result = (byte[]) MMRUtil.extractVectorFromHit(source, "embedding", "doc-1", false);

        assertNotNull(result);
        assertEquals(4, result.length);
        assertEquals((byte) 1, result[0]);
        assertEquals((byte) 2, result[1]);
        assertEquals((byte) 3, result[2]);
        assertEquals((byte) 127, result[3]);
    }

    @Test
    public void testExtractVectorFromHit_NestedPath() {
        Map<String, Object> source = Map.of("user", Map.of("profile", Map.of("embedding", List.of(1.0, 2.0, 3.0))));

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "user.profile.embedding", "doc-2", true);

        assertNotNull(result);
        assertEquals(3, result.length);
        assertEquals(1.0f, result[0], 0.0001f);
        assertEquals(2.0f, result[1], 0.0001f);
        assertEquals(3.0f, result[2], 0.0001f);
    }

    @Test
    public void testExtractVectorFromHit_DeepNestedPath() {
        Map<String, Object> source = Map.of("level1", Map.of("level2", Map.of("level3", Map.of("vector", List.of(5.5, 6.6)))));

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "level1.level2.level3.vector", "doc-3", true);

        assertNotNull(result);
        assertEquals(2, result.length);
        assertEquals(5.5f, result[0], 0.0001f);
        assertEquals(6.6f, result[1], 0.0001f);
    }

    @Test
    public void testExtractVectorFromHit_EmptyVector() {
        Map<String, Object> source = Map.of("embedding", List.of());

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "embedding", "doc-4", true);

        assertNotNull(result);
        assertEquals(0, result.length);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testExtractVectorFromHit_NullSource() {
        MMRUtil.extractVectorFromHit(null, "embedding", "doc-5", true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testExtractVectorFromHit_NullFieldPath() {
        Map<String, Object> source = Map.of("embedding", List.of(1.0, 2.0));
        MMRUtil.extractVectorFromHit(source, null, "doc-6", true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testExtractVectorFromHit_EmptyFieldPath() {
        Map<String, Object> source = Map.of("embedding", List.of(1.0, 2.0));
        MMRUtil.extractVectorFromHit(source, "", "doc-7", true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testExtractVectorFromHit_MissingField() {
        Map<String, Object> source = Map.of("other_field", "value");
        MMRUtil.extractVectorFromHit(source, "embedding", "doc-8", true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testExtractVectorFromHit_MissingNestedField() {
        Map<String, Object> source = Map.of("user", Map.of("name", "John"));
        MMRUtil.extractVectorFromHit(source, "user.profile.embedding", "doc-9", true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testExtractVectorFromHit_NonMapInPath() {
        Map<String, Object> source = Map.of("user", "not_a_map");
        MMRUtil.extractVectorFromHit(source, "user.embedding", "doc-10", true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testExtractVectorFromHit_NonListAtEnd() {
        Map<String, Object> source = Map.of("embedding", "not_a_vector");
        MMRUtil.extractVectorFromHit(source, "embedding", "doc-11", true);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testExtractVectorFromHit_InvalidVectorContent() {
        Map<String, Object> source = Map.of("embedding", List.of("not", "numbers"));
        MMRUtil.extractVectorFromHit(source, "embedding", "doc-12", true);
    }

    // ========== Tests for getMMRFieldMappingByPath ==========

    @Test
    public void testGetMMRFieldMappingByPath_SimpleField() {
        Map<String, Object> mappings = Map.of(
            "properties",
            Map.of("embedding", Map.of("type", "knn_vector", "dimension", 128, "space_type", "l2"))
        );

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "embedding");

        assertNotNull(result);
        assertEquals("knn_vector", result.get("type"));
        assertEquals(128, result.get("dimension"));
        assertEquals("l2", result.get("space_type"));
    }

    @Test
    public void testGetMMRFieldMappingByPath_NestedField() {
        Map<String, Object> mappings = Map.of(
            "properties",
            Map.of(
                "user",
                Map.of(
                    "properties",
                    Map.of("profile", Map.of("properties", Map.of("embedding", Map.of("type", "knn_vector", "dimension", 256))))
                )
            )
        );

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "user.profile.embedding");

        assertNotNull(result);
        assertEquals("knn_vector", result.get("type"));
        assertEquals(256, result.get("dimension"));
    }

    @Test
    public void testGetMMRFieldMappingByPath_NonKnnField() {
        Map<String, Object> mappings = Map.of("properties", Map.of("title", Map.of("type", "text")));

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "title");

        assertNotNull(result);
        assertEquals("text", result.get("type"));
    }

    @Test
    public void testGetMMRFieldMappingByPath_MissingField() {
        Map<String, Object> mappings = Map.of("properties", Map.of("other_field", Map.of("type", "text")));

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "embedding");

        assertNull(result);
    }

    @Test
    public void testGetMMRFieldMappingByPath_NullMappings() {
        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(null, "embedding");
        assertNull(result);
    }

    @Test
    public void testGetMMRFieldMappingByPath_NoProperties() {
        Map<String, Object> mappings = Map.of("settings", Map.of("index", "value"));

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "embedding");

        assertNull(result);
    }

    @Test
    public void testGetMMRFieldMappingByPath_PropertiesNotMap() {
        Map<String, Object> mappings = Map.of("properties", "not_a_map");

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "embedding");

        assertNull(result);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGetMMRFieldMappingByPath_NestedFieldType() {
        Map<String, Object> mappings = Map.of(
            "properties",
            Map.of(
                "nested_field",
                Map.of(
                    "type",
                    ObjectMapper.NESTED_CONTENT_TYPE,
                    "properties",
                    Map.of("embedding", Map.of("type", "knn_vector", "dimension", 128))
                )
            )
        );

        // Should throw IllegalArgumentException because nested fields are not supported
        MMRUtil.getMMRFieldMappingByPath(mappings, "nested_field.embedding");
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGetMMRFieldMappingByPath_NestedInMiddleOfPath() {
        Map<String, Object> mappings = Map.of(
            "properties",
            Map.of(
                "level1",
                Map.of(
                    "properties",
                    Map.of(
                        "nested_level",
                        Map.of("type", ObjectMapper.NESTED_CONTENT_TYPE, "properties", Map.of("embedding", Map.of("type", "knn_vector")))
                    )
                )
            )
        );

        MMRUtil.getMMRFieldMappingByPath(mappings, "level1.nested_level.embedding");
    }

    @Test
    public void testGetMMRFieldMappingByPath_ObjectFieldType() {
        // Object type (not nested) should be allowed
        Map<String, Object> mappings = Map.of(
            "properties",
            Map.of("user", Map.of("type", "object", "properties", Map.of("embedding", Map.of("type", "knn_vector", "dimension", 64))))
        );

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "user.embedding");

        assertNotNull(result);
        assertEquals("knn_vector", result.get("type"));
        assertEquals(64, result.get("dimension"));
    }

    @Test
    public void testGetMMRFieldMappingByPath_FieldWithoutType() {
        // Field without explicit type (implicit object)
        Map<String, Object> mappings = Map.of(
            "properties",
            Map.of("user", Map.of("properties", Map.of("embedding", Map.of("type", "knn_vector"))))
        );

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "user.embedding");

        assertNotNull(result);
        assertEquals("knn_vector", result.get("type"));
    }

    // ========== Edge case and integration tests ==========

    @Test
    public void testExtractVectorFromHit_LargeVector() {
        List<Double> largeVector = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            largeVector.add((double) i);
        }

        Map<String, Object> source = Map.of("embedding", largeVector);

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "embedding", "doc-large", true);

        assertNotNull(result);
        assertEquals(1000, result.length);
        for (int i = 0; i < 1000; i++) {
            assertEquals((float) i, result[i], 0.0001f);
        }
    }

    @Test
    public void testExtractVectorFromHit_NegativeValues() {
        Map<String, Object> source = Map.of("embedding", List.of(-1.5, -2.5, -3.5));

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "embedding", "doc-negative", true);

        assertNotNull(result);
        assertEquals(3, result.length);
        assertEquals(-1.5f, result[0], 0.0001f);
        assertEquals(-2.5f, result[1], 0.0001f);
        assertEquals(-3.5f, result[2], 0.0001f);
    }

    @Test
    public void testExtractVectorFromHit_MixedPositiveNegative() {
        Map<String, Object> source = Map.of("embedding", List.of(-1.0, 0.0, 1.0, -0.5, 0.5));

        float[] result = (float[]) MMRUtil.extractVectorFromHit(source, "embedding", "doc-mixed", true);

        assertNotNull(result);
        assertEquals(5, result.length);
        assertEquals(-1.0f, result[0], 0.0001f);
        assertEquals(0.0f, result[1], 0.0001f);
        assertEquals(1.0f, result[2], 0.0001f);
        assertEquals(-0.5f, result[3], 0.0001f);
        assertEquals(0.5f, result[4], 0.0001f);
    }

    @Test
    public void testGetMMRFieldMappingByPath_ComplexMapping() {
        Map<String, Object> mappings = new HashMap<>();
        Map<String, Object> properties = new HashMap<>();

        Map<String, Object> userField = new HashMap<>();
        Map<String, Object> userProperties = new HashMap<>();

        Map<String, Object> profileField = new HashMap<>();
        Map<String, Object> profileProperties = new HashMap<>();

        Map<String, Object> embeddingField = new HashMap<>();
        embeddingField.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        embeddingField.put("dimension", 512);
        embeddingField.put("space_type", "cosinesimil");

        profileProperties.put("embedding", embeddingField);
        profileField.put("properties", profileProperties);

        userProperties.put("profile", profileField);
        userField.put("properties", userProperties);

        properties.put("user", userField);
        mappings.put("properties", properties);

        Map<String, Object> result = MMRUtil.getMMRFieldMappingByPath(mappings, "user.profile.embedding");

        assertNotNull(result);
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, result.get("type"));
        assertEquals(512, result.get("dimension"));
        assertEquals("cosinesimil", result.get("space_type"));
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;

import org.junit.Before;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.DerivedSourceTestCase;
import org.opensearch.knn.DerivedSourceUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;

import java.util.*;
import java.io.IOException;

import static org.hamcrest.Matchers.containsString;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.DerivedSourceUtils.*;

/**
 * Integration tests for derived source feature for vector fields. Currently, with derived source, there are
 * a few gaps in functionality. Ignoring tests for now as feature is experimental.
 */
public class DerivedSourceIT extends DerivedSourceTestCase {

    private final String snapshot = "snapshot-test";
    private final String repository = "repo";

    @Before
    @SneakyThrows
    public void setUp() {
        super.setUp();
        final String pathRepo = System.getProperty("tests.path.repo");
        Settings repoSettings = Settings.builder().put("compress", randomBoolean()).put("location", pathRepo).build();
        registerRepository(repository, "fs", true, repoSettings);
    }

    @SneakyThrows
    public void testFlatFields() {
        try {
            List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getFlatIndexContexts("derivedit", true, true);
            testDerivedSourceE2E(indexConfigContexts);
        } catch (Exception excp) {
            // TODO: Byte vectors is not supported. Remove the catch checks once BQ (Binary quantization) support is added to Jvector
            // plugin.
            assertTrue(excp.getMessage().contains("validation_exception"));
            assertTrue(excp.getMessage().contains("\\\"disk_ann\\\" is not supported for vector data type \\\"BYTE\\\""));

            // test should do this, but calling it defensively to avoid leaving indices behind on failures.
            super.tearDown();
        }

    }

    @SneakyThrows
    public void testFlatFieldsWithCore() {
        try {
            List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getFlatIndexContexts("derivedit", true, false, true);
            testDerivedSourceE2E(indexConfigContexts);
        } catch (Exception excp) {
            // TODO: Byte vectors is not supported. Remove the catch checks once BQ (Binary quantization) support is added to Jvector
            // plugin.
            assertTrue(excp.getMessage().contains("validation_exception"));
            assertTrue(excp.getMessage().contains("\\\"disk_ann\\\" is not supported for vector data type \\\"BYTE\\\""));

            // test should do this, but calling it defensively to avoid leaving indices behind on failures.
            super.tearDown();
        }
    }

    public void testMetaFieldsWithKnn() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getIndexContextsWithMetaFields("derivedit", true, true);
        testMetaFields(indexConfigContexts);
    }

    public void testMetaFieldsWithCore() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getIndexContextsWithMetaFields("derivedit", true, false, true);
        testMetaFields(indexConfigContexts);
    }

    @SneakyThrows
    public void testObjectField() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getObjectIndexContexts("derivedit", true);
        testDerivedSourceE2E(indexConfigContexts);
    }

    @SneakyThrows
    public void testNestedField() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getNestedIndexContexts("derivedit", true, true);
        testDerivedSourceE2E(indexConfigContexts);
    }

    @SneakyThrows
    public void testNestedFieldWithCore() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getNestedIndexContexts("derivedit", true, false, true);
        expectThrows(
            ResponseException.class,
            () -> createKnnIndex(
                indexConfigContexts.get(0).indexName,
                indexConfigContexts.get(0).getSettings(),
                indexConfigContexts.get(0).getMapping()
            )
        );
    }

    @SneakyThrows
    public void testNestedFieldWithNullables() {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getNestedIndexContexts("derivedit", true, true);
        assertEquals(6, indexConfigContexts.size());

        assertTrue(1 < indexConfigContexts.size());
        DerivedSourceUtils.IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        DerivedSourceUtils.IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);

        createKnnIndex(
            derivedSourceEnabledContext.indexName,
            derivedSourceEnabledContext.getSettings(),
            derivedSourceEnabledContext.getMapping()
        );
        createKnnIndex(
            derivedSourceDisabledContext.indexName,
            derivedSourceDisabledContext.getSettings(),
            derivedSourceDisabledContext.getMapping()
        );
        // make sure that both index has same routing settings
        assertEquals(derivedSourceEnabledContext.isRoutingEnabled, derivedSourceDisabledContext.isRoutingEnabled);

        // Build all docs with null fields
        for (int i = 0; i < derivedSourceDisabledContext.docCount; i++) {
            String doc1 = "{}";
            String doc2 = "{}";

            // using doc id as routing value, which is default
            addKnnDoc(
                derivedSourceEnabledContext.getIndexName(),
                String.valueOf(i + 1),
                doc1,
                derivedSourceEnabledContext.isRoutingEnabled ? String.valueOf(i + 1) : null
            );
            addKnnDoc(
                derivedSourceDisabledContext.getIndexName(),
                String.valueOf(i + 1),
                doc2,
                derivedSourceDisabledContext.isRoutingEnabled ? String.valueOf(i + 1) : null
            );
        }
        refreshAllIndices();

        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            derivedSourceDisabledContext.indexName,
            derivedSourceEnabledContext.indexName,
            derivedSourceEnabledContext.isRoutingEnabled
        );

        // Update all docs with random field vectors
        for (int i = 0; i < derivedSourceDisabledContext.docCount; i++) {
            String doc1 = derivedSourceEnabledContext.buildDoc();
            String doc2 = derivedSourceDisabledContext.buildDoc();

            assertEquals(doc1, doc2);
            // using doc id as routing value, which is default
            updateKnnDoc(
                derivedSourceEnabledContext.getIndexName(),
                String.valueOf(i + 1),
                doc1,
                derivedSourceEnabledContext.isRoutingEnabled ? String.valueOf(i + 1) : null
            );
            updateKnnDoc(
                derivedSourceDisabledContext.getIndexName(),
                String.valueOf(i + 1),
                doc2,
                derivedSourceDisabledContext.isRoutingEnabled ? String.valueOf(i + 1) : null
            );
        }
        refreshAllIndices();

        assertDocsMatch(
            derivedSourceDisabledContext.docCount,
            derivedSourceDisabledContext.indexName,
            derivedSourceEnabledContext.indexName,
            derivedSourceEnabledContext.isRoutingEnabled
        );
    }

    @SneakyThrows
    public void testDerivedSource_whenSegrepLocal_thenDisabled() {
        // Set the data type input for float fields as byte. If derived source gets enabled, the original and derived
        // wont match because original will have source like [0, 1, 2] and derived will have [0.0, 1.0, 2.0]
        final List<Tuple<String, Boolean>> indexPrefixToEnabled = List.of(
            new Tuple<>("original-enable-", true),
            new Tuple<>("original-disable-", false)
        );
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = new ArrayList<>();
        long consistentRandomSeed = random().nextLong();
        for (Tuple<String, Boolean> index : indexPrefixToEnabled) {
            Random random = new Random(consistentRandomSeed);
            DerivedSourceUtils.IndexConfigContext indexConfigContext = DerivedSourceUtils.IndexConfigContext.builder()
                .indexName(getIndexName("deriveit", index.v1(), false))
                .derivedEnabled(index.v2())
                .random(random)
                .settings(index.v2() ? DERIVED_ENABLED_WITH_SEGREP_SETTINGS : null)
                .fields(
                    List.of(
                        DerivedSourceUtils.NestedFieldContext.builder()
                            .fieldPath("nested_1")
                            .children(
                                List.of(
                                    DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                        .fieldPath("nested_1.test_vector")
                                        .dimension(TEST_DIMENSION)
                                        .valueSupplier(randomVectorSupplier(random, TEST_DIMENSION, VectorDataType.FLOAT))
                                        .build()
                                )
                            )
                            .build(),
                        DerivedSourceUtils.NestedFieldContext.builder()
                            .fieldPath("nested_2")
                            .children(
                                List.of(
                                    DerivedSourceUtils.TextFieldType.builder().fieldPath("nested_2.test-text").build(),
                                    DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                        .fieldPath("nested_2.test_vector")
                                        .dimension(TEST_DIMENSION)
                                        .valueSupplier(randomVectorSupplier(random, TEST_DIMENSION, VectorDataType.FLOAT))
                                        .build(),
                                    DerivedSourceUtils.NestedFieldContext.builder()
                                        .fieldPath("nested_2.nested_3")
                                        .children(
                                            List.of(
                                                DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                                                    .fieldPath("nested_2.nested_3.test_vector")
                                                    .dimension(TEST_DIMENSION)
                                                    .valueSupplier(randomVectorSupplier(random, TEST_DIMENSION, VectorDataType.FLOAT))
                                                    .build(),
                                                DerivedSourceUtils.IntFieldType.builder().fieldPath("nested_2.nested_3.test-int").build()
                                            )
                                        )
                                        .build()
                                )
                            )
                            .build(),
                        DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                            .dimension(TEST_DIMENSION)
                            .valueSupplier(randomVectorSupplier(random, TEST_DIMENSION, VectorDataType.FLOAT))
                            .fieldPath("test_vector")
                            .build(),
                        DerivedSourceUtils.TextFieldType.builder().fieldPath("test-text").build(),
                        DerivedSourceUtils.IntFieldType.builder().fieldPath("test-int").build()
                    )
                )
                .build();
            indexConfigContext.init();
            indexConfigContexts.add(indexConfigContext);
        }

        prepareOriginalIndices(indexConfigContexts);
    }

    /**
     * Tests that kNN handles bad documents the same when derived source is enabled and disabled.
     * @throws IOException
     */
    public void testDerivedSource_HandlesInvalidDocuments() throws IOException {
        List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts = getCustomAnalyzerIndexContexts("derivedit", true, true);

        assertTrue(1 < indexConfigContexts.size());
        DerivedSourceUtils.IndexConfigContext derivedSourceEnabledContext = indexConfigContexts.get(0);
        DerivedSourceUtils.IndexConfigContext derivedSourceDisabledContext = indexConfigContexts.get(1);
        createKnnIndex(
            derivedSourceEnabledContext.indexName,
            derivedSourceEnabledContext.getSettings(),
            derivedSourceEnabledContext.getMapping()
        );
        createKnnIndex(
            derivedSourceDisabledContext.indexName,
            derivedSourceDisabledContext.getSettings(),
            derivedSourceDisabledContext.getMapping()
        );
        for (int i = 0; i < derivedSourceDisabledContext.docCount; i++) {
            String doc1 = derivedSourceEnabledContext.buildDoc();
            String doc2 = derivedSourceDisabledContext.buildDoc();
            assertEquals(doc1, doc2);
            boolean dsEnabledException = false;
            boolean dsDisabledException = false;
            try {
                addKnnDoc(derivedSourceEnabledContext.getIndexName(), String.valueOf(i + 1), doc1);
            } catch (ResponseException e) {
                assertTrue(e.getMessage().contains("number_format_exception"));
                dsEnabledException = true;
            }
            try {
                addKnnDoc(derivedSourceDisabledContext.getIndexName(), String.valueOf(i + 1), doc2);
            } catch (ResponseException e) {
                assertTrue(e.getMessage().contains("number_format_exception"));
                dsDisabledException = true;
            }
            assertEquals(dsEnabledException, dsDisabledException);
        }
    }

    private void testMetaFields(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts) {
        List<String> metaFields = List.of(ROUTING_FIELD, "_id", "_score");

        assertEquals("Expected 6 index contexts for meta fields test", 6, indexConfigContexts.size());
        prepareOriginalIndices(indexConfigContexts);

        List<Object> searchResults = testSearch(indexConfigContexts);
        assertFalse("Search results should not be empty", searchResults.isEmpty());

        for (int i = 0; i < searchResults.size(); i++) {
            Object searchResult = searchResults.get(i);
            assertNotNull("Search result at index " + i + " should not be null", searchResult);

            Map<String, Object> hits = (Map<String, Object>) searchResult;
            for (String metaField : metaFields) {
                assertTrue(String.format("Missing meta field '%s' in search result %d", metaField, i), hits.containsKey(metaField));
                assertNotNull(
                    String.format("Meta field '%s' value should not be null in search result %d", metaField, i),
                    hits.get(metaField)
                );
            }
        }
    }

    /**
     * Single method for running end to end tests for different index configurations for derived source. In general,
     * flow of operations are
     *
     * @param indexConfigContexts {@link DerivedSourceUtils.IndexConfigContext}
     */
    @SneakyThrows
    private void testDerivedSourceE2E(List<DerivedSourceUtils.IndexConfigContext> indexConfigContexts) {
        assertEquals(6, indexConfigContexts.size());

        // Prepare the indices by creating them and ingesting data into them
        prepareOriginalIndices(indexConfigContexts);

        // Merging
        testMerging(indexConfigContexts);

        // Update. Skipping update tests for nested docs for now. Will add in the future.
        testUpdate(indexConfigContexts);

        // Delete
        testDelete(indexConfigContexts);

        // Search
        testSearch(indexConfigContexts);

        // Reindex
        testReindex(indexConfigContexts);

        // Snapshot restore
        testSnapshotRestore(repository, snapshot + getTestName().toLowerCase(Locale.ROOT), indexConfigContexts);
    }

    @SneakyThrows
    public void testDefaultSetting() {
        String indexName = getIndexName("defaults", "test", false);
        String fieldName = "test";
        String indexNameDisabled = "disabled";
        int dimension = 16;
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .endObject()
            .endObject()
            .endObject();
        String mapping = builder.toString();
        createKnnIndex(indexName, mapping);
        validateDerivedSetting(indexName, true);
        createIndex(indexNameDisabled, Settings.builder().build());
        validateDerivedSetting(indexNameDisabled, false);
    }

    @SneakyThrows
    public void testBlockSettingIfKNNFalse() {
        String indexName = getIndexName("setting-blocked", "test", false);
        String fieldName = "test";
        int dimension = 16;
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(fieldName)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .endObject()
            .endObject()
            .endObject();
        String mapping = builder.toString();
        expectThrows(
            ResponseException.class,
            () -> createKnnIndex(
                indexName,
                Settings.builder().put("index.knn", false).put("index.knn.derived_source.enabled", true).build(),
                mapping
            )
        );

        expectThrows(
            ResponseException.class,
            () -> createKnnIndex(indexName, Settings.builder().put("index.knn.derived_source.enabled", true).build(), mapping)
        );
    }

    @SuppressWarnings("unchecked")
    private List<Float> extractVector(Map<String, Object> source, String... path) {
        Object current = source;
        for (String key : path) {
            if (current instanceof Map) {
                current = ((Map<String, Object>) current).get(key);
            } else {
                return null;
            }
        }
        if (current instanceof List) {
            return (List<Float>) current;
        }
        return null;
    }

    @SneakyThrows
    public void testSourceFiltering_withVariousIncludeExcludeCombinations() {
        String indexName = getIndexName("source-filtering", "combinations", false);
        String VECTOR_FIELD_1 = "test_vector";
        String VECTOR_FIELD_2 = "temp_vector";
        String VECTOR_FIELD_3 = "user_vector";
        String TEXT_FIELD = "description";
        int DIMENSION = 3;

        // Create index with multiple vector fields and a text field
        XContentBuilder mappingBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(KNNConstants.PROPERTIES)
            .startObject(VECTOR_FIELD_1)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(KNNConstants.DIMENSION, DIMENSION)
            .endObject()
            .startObject(VECTOR_FIELD_2)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(KNNConstants.DIMENSION, DIMENSION)
            .endObject()
            .startObject(VECTOR_FIELD_3)
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(KNNConstants.DIMENSION, DIMENSION)
            .endObject()
            .startObject(TEXT_FIELD)
            .field(KNNConstants.TYPE, "text")
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(
            indexName,
            Settings.builder().put("index.knn", true).put("index.knn.derived_source.enabled", true).build(),
            mappingBuilder.toString()
        );

        // Index a document with all fields
        XContentBuilder docBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .array(VECTOR_FIELD_1, 1.0f, 2.0f, 3.0f)
            .array(VECTOR_FIELD_2, 4.0f, 5.0f, 6.0f)
            .array(VECTOR_FIELD_3, 7.0f, 8.0f, 9.0f)
            .field(TEXT_FIELD, "test description")
            .endObject();
        addKnnDoc(indexName, "1", docBuilder.toString());

        refreshIndex(indexName);

        // Test 1: No filtering - all fields returned
        assertSourceFiltering(
            indexName,
            null,  // includes
            null,  // excludes
            new String[] { VECTOR_FIELD_1, VECTOR_FIELD_2, VECTOR_FIELD_3, TEXT_FIELD },  // expected present
            new String[] {}  // expected absent
        );

        // Test 2: Only includes - only specified fields returned
        assertSourceFiltering(
            indexName,
            new String[] { VECTOR_FIELD_1, TEXT_FIELD },
            null,
            new String[] { VECTOR_FIELD_1, TEXT_FIELD },
            new String[] { VECTOR_FIELD_2, VECTOR_FIELD_3 }
        );

        // Test 3: Only excludes - all except specified fields returned
        assertSourceFiltering(
            indexName,
            null,
            new String[] { VECTOR_FIELD_1 },
            new String[] { VECTOR_FIELD_2, VECTOR_FIELD_3, TEXT_FIELD },
            new String[] { VECTOR_FIELD_1 }
        );

        // Test 4: Both includes and excludes - throws IllegalArgumentException because vector field cannot be in both includes and excludes
        ResponseException ex = expectThrows(
            ResponseException.class,
            () -> sourceFiltering(indexName, new String[] { VECTOR_FIELD_1, VECTOR_FIELD_2, TEXT_FIELD }, new String[] { VECTOR_FIELD_2 })
        );
        assertThat(
            ex.getMessage(),
            containsString("The same entry [" + VECTOR_FIELD_2 + "] cannot be both included and excluded in _source.")
        );

        // Test 5: Wildcard includes - only matching fields returned
        assertSourceFiltering(
            indexName,
            new String[] { "t*" },  // matches test_vector, temp_vector
            null,
            new String[] { VECTOR_FIELD_1, VECTOR_FIELD_2 },
            new String[] { VECTOR_FIELD_3, TEXT_FIELD }
        );

        // Test 6: Wildcard excludes - all except matching fields returned
        assertSourceFiltering(
            indexName,
            null,
            new String[] { "t*" },  // excludes test_vector, temp_vector
            new String[] { VECTOR_FIELD_3, TEXT_FIELD },
            new String[] { VECTOR_FIELD_1, VECTOR_FIELD_2 }
        );

        // Test 7: Wildcard includes with specific excludes
        assertSourceFiltering(
            indexName,
            new String[] { "t*", VECTOR_FIELD_3 },  // includes test_vector, temp_vector, user_vector
            new String[] { VECTOR_FIELD_1 },  // excludes test_vector
            new String[] { VECTOR_FIELD_2, VECTOR_FIELD_3 },
            new String[] { VECTOR_FIELD_1, TEXT_FIELD }
        );

        // Test 8: Empty includes array - all fields returned (no filtering)
        assertSourceFiltering(
            indexName,
            new String[] {},
            null,
            new String[] { VECTOR_FIELD_1, VECTOR_FIELD_2, VECTOR_FIELD_3, TEXT_FIELD },
            new String[] {}
        );

        // Test 9: Empty excludes array - all fields returned (no filtering)
        assertSourceFiltering(
            indexName,
            null,
            new String[] {},
            new String[] { VECTOR_FIELD_1, VECTOR_FIELD_2, VECTOR_FIELD_3, TEXT_FIELD },
            new String[] {}
        );
    }

    /**
     * Tests that bulk indexing with dynamic templates works correctly when derived source is enabled.
     * This is a regression test for https://github.com/opensearch-project/k-NN/issues/3012
     *
     * The bug: When bulk indexing documents with dynamic templates that create different field
     * mappings per document, only the
     * first document's vector was correctly reconstructed. Subsequent documents returned the
     * mask value (1) instead of the actual vector.
     *
     * Root cause: Segment attributes were written at segment creation time (in fieldsWriter()),
     * when only the first document's dynamic mapping existed. The fix moves attribute writing
     * to finish(), when all documents have been parsed and all mappings exist.
     */
    @SneakyThrows
    public void testDerivedSource_withDynamicTemplates_andBulkIndexing() {
        String indexName = "test-derived-dynamic-template";
        int dimension = 3;

        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put("index.knn.derived_source.enabled", true)
            .build();

        XContentBuilder mappingsBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startArray("dynamic_templates")
            .startObject()
            .startObject("knn_vector_template")
            .field("path_match", "similar_products_vector.*.clip_vit_base_patch32")
            .startObject("mapping")
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .startObject("method")
            .field("engine", "lucene")
            .field("space_type", "l2")
            .field("name", "hnsw")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endArray()
            .endObject();

        createKnnIndex(indexName, settings, mappingsBuilder.toString());

        StringBuilder bulkRequestBody = new StringBuilder();

        bulkRequestBody.append("{\"index\": {\"_index\": \"").append(indexName).append("\", \"_id\": \"doc1\"}}\n");
        bulkRequestBody.append("{\"similar_products_vector\": {\"key_1001\": {\"clip_vit_base_patch32\": [1.0, 1.0, 1.0]}}}\n");

        bulkRequestBody.append("{\"index\": {\"_index\": \"").append(indexName).append("\", \"_id\": \"doc2\"}}\n");
        bulkRequestBody.append("{\"similar_products_vector\": {\"key_1002\": {\"clip_vit_base_patch32\": [2.0, 2.0, 2.0]}}}\n");

        bulkRequestBody.append("{\"index\": {\"_index\": \"").append(indexName).append("\", \"_id\": \"doc3\"}}\n");
        bulkRequestBody.append("{\"similar_products_vector\": {\"key_1003\": {\"clip_vit_base_patch32\": [3.0, 3.0, 3.0]}}}\n");

        Request bulkRequest = new Request("POST", "/_bulk");
        bulkRequest.setJsonEntity(bulkRequestBody.toString());
        Response bulkResponse = client().performRequest(bulkRequest);
        assertEquals(RestStatus.OK.getStatus(), bulkResponse.getStatusLine().getStatusCode());

        refreshIndex(indexName);

        Map<String, Object> doc1Source = getKnnDoc(indexName, "doc1");
        Map<String, Object> doc2Source = getKnnDoc(indexName, "doc2");
        Map<String, Object> doc3Source = getKnnDoc(indexName, "doc3");

        List<Float> retrievedVector1 = extractVector(doc1Source, "similar_products_vector", "key_1001", "clip_vit_base_patch32");
        List<Float> retrievedVector2 = extractVector(doc2Source, "similar_products_vector", "key_1002", "clip_vit_base_patch32");
        List<Float> retrievedVector3 = extractVector(doc3Source, "similar_products_vector", "key_1003", "clip_vit_base_patch32");

        assertNotNull("Vector 1 should not be null - got mask value instead of array", retrievedVector1);
        assertNotNull("Vector 2 should not be null - got mask value instead of array", retrievedVector2);
        assertNotNull("Vector 3 should not be null - got mask value instead of array", retrievedVector3);

        assertEquals("Vector 1 should have correct dimension", dimension, retrievedVector1.size());
        assertEquals("Vector 2 should have correct dimension", dimension, retrievedVector2.size());
        assertEquals("Vector 3 should have correct dimension", dimension, retrievedVector3.size());

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testDerivedSource_withMixedCaseObjectVectorField() {
        String indexName = getIndexName("derived-source", "mixed-case", false);
        int dimension = 3;

        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put("index.knn.derived_source.enabled", true)
            .build();

        XContentBuilder mappingBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(KNNConstants.PROPERTIES)
            .startObject("vectorSearch")
            .startObject(KNNConstants.PROPERTIES)
            .startObject("nameVector")
            .field(KNNConstants.TYPE, KNNConstants.TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension)
            .startObject("method")
            .field("engine", "lucene")
            .field("space_type", "l2")
            .field("name", "hnsw")
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, settings, mappingBuilder.toString());

        XContentBuilder docBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("vectorSearch")
            .array("nameVector", 1.0f, 2.0f, 3.0f)
            .endObject()
            .endObject();

        addKnnDoc(indexName, "1", docBuilder.toString());
        refreshIndex(indexName);

        List<?> retrievedVector = extractVector(getKnnDoc(indexName, "1"), "vectorSearch", "nameVector");

        assertNotNull("Mixed-case vector field should be reconstructed instead of returning the mask value", retrievedVector);
        assertEquals(dimension, retrievedVector.size());
        assertEquals(1.0f, ((Number) retrievedVector.get(0)).floatValue(), 0.0f);
        assertEquals(2.0f, ((Number) retrievedVector.get(1)).floatValue(), 0.0f);
        assertEquals(3.0f, ((Number) retrievedVector.get(2)).floatValue(), 0.0f);

        deleteKNNIndex(indexName);
    }
}

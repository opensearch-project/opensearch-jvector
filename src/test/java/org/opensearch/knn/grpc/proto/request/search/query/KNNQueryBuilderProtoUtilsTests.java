/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.grpc.proto.request.search.query;

import org.junit.Before;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.protobufs.KnnQuery;
import org.opensearch.protobufs.KnnQueryRescore;
import org.opensearch.protobufs.ObjectMap;
import org.opensearch.protobufs.QueryContainer;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.transport.grpc.spi.QueryBuilderProtoConverterRegistry;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Unit tests for KNNQueryBuilderProtoUtils.
 * Tests conversion from Protocol Buffer KnnQuery to KNNQueryBuilder.
 */
public class KNNQueryBuilderProtoUtilsTests extends OpenSearchTestCase {

    private QueryBuilderProtoConverterRegistry mockRegistry;
    private QueryBuilder mockQueryBuilder;

    @Before
    public void setup() {
        mockRegistry = mock(QueryBuilderProtoConverterRegistry.class);
        mockQueryBuilder = mock(QueryBuilder.class);
    }

    /**
     * Test basic conversion with field, vector, and k.
     */
    public void testFromProto_BasicQuery() {
        KnnQuery knnQuery = KnnQuery.newBuilder().setField("test_field").addVector(1.0f).addVector(2.0f).addVector(3.0f).setK(5).build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery, mockRegistry);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertEquals("test_field", knnQueryBuilder.fieldName());
        assertArrayEquals(new float[] { 1.0f, 2.0f, 3.0f }, (float[]) knnQueryBuilder.vector(), 0.001f);
        assertEquals((Object) Integer.valueOf(5), (Object) knnQueryBuilder.getK());
    }

    /**
     * Test conversion with boost.
     */
    public void testFromProto_WithBoost() {
        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .addVector(3.0f)
            .setK(5)
            .setBoost(2.5f)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery, mockRegistry);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertEquals(2.5f, knnQueryBuilder.boost(), 0.001f);
    }

    /**
     * Test conversion with query name.
     */
    public void testFromProto_WithQueryName() {
        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .setK(5)
            .setXName("my_knn_query")
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery, mockRegistry);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertEquals("my_knn_query", knnQueryBuilder.queryName());
    }

    /**
     * Test conversion with maxDistance.
     */
    public void testFromProto_WithMaxDistance() {
        KnnQuery knnQuery = KnnQuery.newBuilder().setField("test_field").addVector(1.0f).addVector(2.0f).setMaxDistance(0.5f).build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery, mockRegistry);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertEquals(0.5f, knnQueryBuilder.getMaxDistance(), 0.001f);
    }

    /**
     * Test conversion with minScore.
     */
    public void testFromProto_WithMinScore() {
        KnnQuery knnQuery = KnnQuery.newBuilder().setField("test_field").addVector(1.0f).addVector(2.0f).setMinScore(0.8f).build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery, mockRegistry);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertEquals(0.8f, knnQueryBuilder.getMinScore(), 0.001f);
    }

    /**
     * Test conversion with method parameters using ObjectMap.
     */
    public void testFromProto_WithMethodParameters() {
        ObjectMap.Value efSearchValue = ObjectMap.Value.newBuilder().setInt32(100).build();
        ObjectMap.Value nprobesValue = ObjectMap.Value.newBuilder().setInt32(50).build();

        ObjectMap methodParams = ObjectMap.newBuilder().putFields("ef_search", efSearchValue).putFields("nprobes", nprobesValue).build();

        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .setK(5)
            .setMethodParameters(methodParams)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery, mockRegistry);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertNotNull(knnQueryBuilder.getMethodParameters());
        assertEquals(100, knnQueryBuilder.getMethodParameters().get("ef_search"));
        assertEquals(50, knnQueryBuilder.getMethodParameters().get("nprobes"));
    }

    /**
     * Test ObjectMap value conversion for different types.
     */
    public void testConvertObjectMapValue_AllTypes() {
        // Test int32
        ObjectMap.Value int32Val = ObjectMap.Value.newBuilder().setInt32(42).build();
        assertEquals(42, KNNQueryBuilderProtoUtils.convertObjectMapValue(int32Val));

        // Test int64
        ObjectMap.Value int64Val = ObjectMap.Value.newBuilder().setInt64(123456789L).build();
        assertEquals(123456789L, KNNQueryBuilderProtoUtils.convertObjectMapValue(int64Val));

        // Test float
        ObjectMap.Value floatVal = ObjectMap.Value.newBuilder().setFloat(3.14f).build();
        assertEquals(3.14f, (Float) KNNQueryBuilderProtoUtils.convertObjectMapValue(floatVal), 0.001f);

        // Test double
        ObjectMap.Value doubleVal = ObjectMap.Value.newBuilder().setDouble(2.718).build();
        assertEquals(2.718, (Double) KNNQueryBuilderProtoUtils.convertObjectMapValue(doubleVal), 0.001);

        // Test string
        ObjectMap.Value stringVal = ObjectMap.Value.newBuilder().setString("test").build();
        assertEquals("test", KNNQueryBuilderProtoUtils.convertObjectMapValue(stringVal));

        // Test boolean
        ObjectMap.Value boolVal = ObjectMap.Value.newBuilder().setBool(true).build();
        assertEquals(true, KNNQueryBuilderProtoUtils.convertObjectMapValue(boolVal));
    }

    /**
     * Test conversion with filter query.
     */
    public void testFromProto_WithFilter() {
        QueryContainer filterContainer = QueryContainer.newBuilder().build();
        TermQueryBuilder termQuery = new TermQueryBuilder("status", "active");

        when(mockRegistry.fromProto(any(QueryContainer.class))).thenReturn(termQuery);

        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .setK(5)
            .setFilter(filterContainer)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery, mockRegistry);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertNotNull(knnQueryBuilder.getFilter());
    }

    /**
     * Test conversion with rescore enabled.
     */
    public void testFromProto_WithRescoreEnabled() {
        KnnQueryRescore rescore = KnnQueryRescore.newBuilder().setEnable(true).build();

        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .setK(5)
            .setRescore(rescore)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery, mockRegistry);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertNotNull(knnQueryBuilder.getRescoreContext());
        assertEquals(RescoreContext.getDefault(), knnQueryBuilder.getRescoreContext());
    }

    /**
     * Test conversion with rescore disabled.
     */
    public void testFromProto_WithRescoreDisabled() {
        KnnQueryRescore rescore = KnnQueryRescore.newBuilder().setEnable(false).build();

        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .setK(5)
            .setRescore(rescore)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery, mockRegistry);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertEquals(RescoreContext.EXPLICITLY_DISABLED_RESCORE_CONTEXT, knnQueryBuilder.getRescoreContext());
    }

    /**
     * Test conversion with rescore context (oversample factor).
     */
    public void testFromProto_WithRescoreContext() {
        org.opensearch.protobufs.RescoreContext rescoreCtx = org.opensearch.protobufs.RescoreContext.newBuilder()
            .setOversampleFactor(2.0f)
            .build();

        KnnQueryRescore rescore = KnnQueryRescore.newBuilder().setContext(rescoreCtx).build();

        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .setK(5)
            .setRescore(rescore)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery, mockRegistry);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertNotNull(knnQueryBuilder.getRescoreContext());
        assertEquals(2.0f, knnQueryBuilder.getRescoreContext().getOversampleFactor(), 0.001f);
    }

    /**
     * Test conversion with expandNested.
     */
    public void testFromProto_WithExpandNested() {
        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("test_field")
            .addVector(1.0f)
            .addVector(2.0f)
            .setK(5)
            .setExpandNestedDocs(true)
            .build();

        QueryBuilder result = KNNQueryBuilderProtoUtils.fromProto(knnQuery, mockRegistry);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertTrue(knnQueryBuilder.getExpandNested());
    }

    /**
     * Test error handling for missing field.
     */
    public void testFromProto_MissingField_ThrowsException() {
        KnnQuery knnQuery = KnnQuery.newBuilder().addVector(1.0f).addVector(2.0f).setK(5).build();

        expectThrows(IllegalArgumentException.class, () -> { KNNQueryBuilderProtoUtils.fromProto(knnQuery, mockRegistry); });
    }

    /**
     * Test error handling for empty vector.
     */
    public void testFromProto_EmptyVector_ThrowsException() {
        KnnQuery knnQuery = KnnQuery.newBuilder().setField("test_field").setK(5).build();

        expectThrows(IllegalArgumentException.class, () -> { KNNQueryBuilderProtoUtils.fromProto(knnQuery, mockRegistry); });
    }
}

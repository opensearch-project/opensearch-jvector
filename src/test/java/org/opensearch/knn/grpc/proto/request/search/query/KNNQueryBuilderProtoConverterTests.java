/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.grpc.proto.request.search.query;

import org.junit.Before;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.protobufs.KnnQuery;
import org.opensearch.protobufs.QueryContainer;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.transport.grpc.spi.QueryBuilderProtoConverterRegistry;

import static org.mockito.Mockito.mock;

/**
 * Unit tests for KNNQueryBuilderProtoConverter.
 * Tests the SPI interface implementation and delegation to Utils.
 */
public class KNNQueryBuilderProtoConverterTests extends OpenSearchTestCase {

    private KNNQueryBuilderProtoConverter converter;
    private QueryBuilderProtoConverterRegistry mockRegistry;

    @Before
    public void setup() {
        converter = new KNNQueryBuilderProtoConverter();
        mockRegistry = mock(QueryBuilderProtoConverterRegistry.class);
    }

    /**
     * Test that registry injection works.
     */
    public void testSetRegistry() {
        converter.setRegistry(mockRegistry);
        // No exception means success - registry is stored internally
    }

    /**
     * Test that getHandledQueryCase returns KNN.
     */
    public void testGetHandledQueryCase() {
        assertEquals(QueryContainer.QueryContainerCase.KNN, converter.getHandledQueryCase());
    }

    /**
     * Test fromProto delegates to Utils correctly.
     */
    public void testFromProto_DelegatesToUtils() {
        converter.setRegistry(mockRegistry);

        KnnQuery knnQuery = KnnQuery.newBuilder().setField("test_field").addVector(1.0f).addVector(2.0f).addVector(3.0f).setK(5).build();

        QueryContainer queryContainer = QueryContainer.newBuilder().setKnn(knnQuery).build();

        QueryBuilder result = converter.fromProto(queryContainer);

        assertNotNull(result);
        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertEquals("test_field", knnQueryBuilder.fieldName());
        assertEquals((Object) Integer.valueOf(5), (Object) knnQueryBuilder.getK());
    }

    /**
     * Test fromProto with all optional fields.
     */
    public void testFromProto_WithAllFields() {
        converter.setRegistry(mockRegistry);

        KnnQuery knnQuery = KnnQuery.newBuilder()
            .setField("vector_field")
            .addVector(0.1f)
            .addVector(0.2f)
            .addVector(0.3f)
            .setK(10)
            .setBoost(2.0f)
            .setXName("test_query")
            .setExpandNestedDocs(true)
            .build();

        QueryContainer queryContainer = QueryContainer.newBuilder().setKnn(knnQuery).build();

        QueryBuilder result = converter.fromProto(queryContainer);

        assertTrue(result instanceof KNNQueryBuilder);
        KNNQueryBuilder knnQueryBuilder = (KNNQueryBuilder) result;
        assertEquals("vector_field", knnQueryBuilder.fieldName());
        assertEquals((Object) Integer.valueOf(10), (Object) knnQueryBuilder.getK());
        assertEquals(2.0f, knnQueryBuilder.boost(), 0.001f);
        assertEquals("test_query", knnQueryBuilder.queryName());
        assertTrue(knnQueryBuilder.getExpandNested());
    }

    /**
     * Test that fromProto throws exception when registry not set.
     */
    public void testFromProto_WithoutRegistry_ThrowsException() {
        // Don't set registry
        KnnQuery knnQuery = KnnQuery.newBuilder().setField("test_field").addVector(1.0f).setK(5).build();

        QueryContainer queryContainer = QueryContainer.newBuilder().setKnn(knnQuery).build();

        expectThrows(IllegalStateException.class, () -> { converter.fromProto(queryContainer); });
    }

    /**
     * Test that fromProto throws exception for wrong query type.
     */
    public void testFromProto_WrongQueryType_ThrowsException() {
        converter.setRegistry(mockRegistry);

        // Create QueryContainer with Term query instead of KNN
        QueryContainer queryContainer = QueryContainer.newBuilder().build();

        expectThrows(IllegalArgumentException.class, () -> { converter.fromProto(queryContainer); });
    }

    /**
     * Test error handling for invalid KNN query (missing field).
     */
    public void testFromProto_InvalidQuery_ThrowsException() {
        converter.setRegistry(mockRegistry);

        KnnQuery knnQuery = KnnQuery.newBuilder().addVector(1.0f).setK(5).build(); // Missing field name

        QueryContainer queryContainer = QueryContainer.newBuilder().setKnn(knnQuery).build();

        expectThrows(IllegalArgumentException.class, () -> { converter.fromProto(queryContainer); });
    }

    /**
     * Test that converter can be instantiated (for SPI).
     */
    public void testInstantiation() {
        KNNQueryBuilderProtoConverter newConverter = new KNNQueryBuilderProtoConverter();
        assertNotNull(newConverter);
        assertEquals(QueryContainer.QueryContainerCase.KNN, newConverter.getHandledQueryCase());
    }
}

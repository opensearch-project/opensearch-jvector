/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.ToChildBlockJoinQuery;
import org.apache.lucene.index.Term;
import org.mockito.MockedConstruction;
import org.mockito.Mockito;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.index.search.NestedHelper;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Tests for BaseQueryFactory - base class for creating vector search queries.
 */
public class BaseQueryFactoryTests extends KNNTestCase {

    private static final String FIELD_NAME = "test_field";
    private static final String INDEX_NAME = "test_index";
    private static final float[] TEST_VECTOR = new float[] { 1.0f, 2.0f, 3.0f };

    public void testCreateQueryRequest_Builder() {
        BaseQueryFactory.CreateQueryRequest request = BaseQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.LUCENE)
            .indexName(INDEX_NAME)
            .fieldName(FIELD_NAME)
            .vector(TEST_VECTOR)
            .vectorDataType(VectorDataType.FLOAT)
            .k(10)
            .build();

        assertEquals(KNNEngine.LUCENE, request.getKnnEngine());
        assertEquals(INDEX_NAME, request.getIndexName());
        assertEquals(FIELD_NAME, request.getFieldName());
        assertArrayEquals(TEST_VECTOR, request.getVector(), 0.0001f);
        assertEquals(VectorDataType.FLOAT, request.getVectorDataType());
        assertEquals(Integer.valueOf(10), request.getK());
    }

    public void testCreateQueryRequest_WithAllFields() {
        QueryBuilder filter = new TermQueryBuilder("field", "value");
        QueryShardContext context = mock(QueryShardContext.class);
        byte[] byteVector = new byte[] { 1, 2, 3 };

        BaseQueryFactory.CreateQueryRequest request = BaseQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.JVECTOR)
            .indexName(INDEX_NAME)
            .fieldName(FIELD_NAME)
            .vector(TEST_VECTOR)
            .byteVector(byteVector)
            .vectorDataType(VectorDataType.BYTE)
            .k(20)
            .radius(0.5f)
            .filter(filter)
            .context(context)
            .expandNested(true)
            .build();

        assertEquals(KNNEngine.JVECTOR, request.getKnnEngine());
        assertEquals(INDEX_NAME, request.getIndexName());
        assertEquals(FIELD_NAME, request.getFieldName());
        assertArrayEquals(TEST_VECTOR, request.getVector(), 0.0001f);
        assertArrayEquals(byteVector, request.getByteVector());
        assertEquals(VectorDataType.BYTE, request.getVectorDataType());
        assertEquals(Integer.valueOf(20), request.getK());
        assertEquals(Float.valueOf(0.5f), request.getRadius());
        assertTrue(request.getFilter().isPresent());
        assertEquals(filter, request.getFilter().get());
        assertTrue(request.getContext().isPresent());
        assertEquals(context, request.getContext().get());
        assertEquals(Boolean.TRUE, request.getExpandNested());
    }

    public void testCreateQueryRequest_OptionalFields() {
        BaseQueryFactory.CreateQueryRequest request = BaseQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.LUCENE)
            .indexName(INDEX_NAME)
            .build();

        assertFalse(request.getFilter().isPresent());
        assertFalse(request.getContext().isPresent());
    }

    public void testGetFilterQuery_WithNoFilter() {
        BaseQueryFactory.CreateQueryRequest request = BaseQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.LUCENE)
            .indexName(INDEX_NAME)
            .build();

        Query filterQuery = BaseQueryFactory.getFilterQuery(request);
        assertNull(filterQuery);
    }

    public void testGetFilterQuery_WithFilter_NoParentFilter() throws IOException {
        QueryBuilder filterBuilder = mock(QueryBuilder.class);
        QueryShardContext mockContext = mock(QueryShardContext.class);
        Query expectedQuery = new TermQuery(new Term("field", "value"));

        when(filterBuilder.toQuery(mockContext)).thenReturn(expectedQuery);
        when(mockContext.getParentFilter()).thenReturn(null);

        BaseQueryFactory.CreateQueryRequest request = BaseQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.LUCENE)
            .indexName(INDEX_NAME)
            .fieldName(FIELD_NAME)
            .filter(filterBuilder)
            .context(mockContext)
            .build();

        Query filterQuery = BaseQueryFactory.getFilterQuery(request);
        assertEquals(expectedQuery, filterQuery);
    }

    public void testGetFilterQuery_WithFilter_WithParentFilter_MightMatchNested() throws IOException {
        QueryBuilder filterBuilder = mock(QueryBuilder.class);
        QueryShardContext mockContext = mock(QueryShardContext.class);
        MapperService mockMapperService = mock(MapperService.class);
        Query expectedQuery = new TermQuery(new Term("field", "value"));
        BitSetProducer parentFilter = mock(BitSetProducer.class);

        when(filterBuilder.toQuery(mockContext)).thenReturn(expectedQuery);
        when(mockContext.getParentFilter()).thenReturn(parentFilter);
        when(mockContext.getMapperService()).thenReturn(mockMapperService);

        try (
            MockedConstruction<NestedHelper> mockedNestedHelper = Mockito.mockConstruction(
                NestedHelper.class,
                (mock, context) -> when(mock.mightMatchNestedDocs(expectedQuery)).thenReturn(true)
            )
        ) {
            BaseQueryFactory.CreateQueryRequest request = BaseQueryFactory.CreateQueryRequest.builder()
                .knnEngine(KNNEngine.LUCENE)
                .indexName(INDEX_NAME)
                .fieldName(FIELD_NAME)
                .filter(filterBuilder)
                .context(mockContext)
                .build();

            Query filterQuery = BaseQueryFactory.getFilterQuery(request);
            assertEquals(expectedQuery, filterQuery);
        }
    }

    public void testGetFilterQuery_WithFilter_WithParentFilter_DoesNotMatchNested() throws IOException {
        QueryBuilder filterBuilder = mock(QueryBuilder.class);
        QueryShardContext mockContext = mock(QueryShardContext.class);
        MapperService mockMapperService = mock(MapperService.class);
        Query expectedQuery = new TermQuery(new Term("field", "value"));
        BitSetProducer parentFilter = mock(BitSetProducer.class);

        when(filterBuilder.toQuery(mockContext)).thenReturn(expectedQuery);
        when(mockContext.getParentFilter()).thenReturn(parentFilter);
        when(mockContext.getMapperService()).thenReturn(mockMapperService);

        try (
            MockedConstruction<NestedHelper> mockedNestedHelper = Mockito.mockConstruction(
                NestedHelper.class,
                (mock, context) -> when(mock.mightMatchNestedDocs(expectedQuery)).thenReturn(false)
            )
        ) {
            BaseQueryFactory.CreateQueryRequest request = BaseQueryFactory.CreateQueryRequest.builder()
                .knnEngine(KNNEngine.LUCENE)
                .indexName(INDEX_NAME)
                .fieldName(FIELD_NAME)
                .filter(filterBuilder)
                .context(mockContext)
                .build();

            Query filterQuery = BaseQueryFactory.getFilterQuery(request);
            assertTrue(filterQuery instanceof ToChildBlockJoinQuery);
        }
    }

    public void testGetFilterQuery_WithoutContext_ThrowsException() {
        QueryBuilder filterBuilder = new TermQueryBuilder("field", "value");

        BaseQueryFactory.CreateQueryRequest request = BaseQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.LUCENE)
            .indexName(INDEX_NAME)
            .fieldName(FIELD_NAME)
            .filter(filterBuilder)
            .build();

        RuntimeException exception = expectThrows(RuntimeException.class, () -> BaseQueryFactory.getFilterQuery(request));
        assertTrue(exception.getMessage().contains("Shard context cannot be null"));
    }

    public void testGetFilterQuery_IOExceptionWrapped() throws IOException {
        QueryBuilder filterBuilder = mock(QueryBuilder.class);
        QueryShardContext mockContext = mock(QueryShardContext.class);

        when(filterBuilder.toQuery(mockContext)).thenThrow(new IOException("Test IO exception"));

        BaseQueryFactory.CreateQueryRequest request = BaseQueryFactory.CreateQueryRequest.builder()
            .knnEngine(KNNEngine.LUCENE)
            .indexName(INDEX_NAME)
            .fieldName(FIELD_NAME)
            .filter(filterBuilder)
            .context(mockContext)
            .build();

        RuntimeException exception = expectThrows(RuntimeException.class, () -> BaseQueryFactory.getFilterQuery(request));
        assertTrue(exception.getMessage().contains("Cannot create query with filter"));
        assertTrue(exception.getCause() instanceof IOException);
    }
}

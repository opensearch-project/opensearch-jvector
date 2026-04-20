/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.collect.ImmutableMap;
import lombok.SneakyThrows;
import org.apache.lucene.search.FloatVectorSimilarityQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.junit.Before;
import org.opensearch.cluster.ClusterModule;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.core.common.io.stream.NamedWriteableRegistry;
import org.opensearch.core.index.Index;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.index.query.QueryRewriteContext;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

import static org.hamcrest.Matchers.instanceOf;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.engine.KNNEngine.ENGINES_SUPPORTING_RADIAL_SEARCH;

public class KNNQueryBuilderTests extends KNNTestCase {

    private static final String FIELD_NAME = "myvector";
    private static final int K = 1;
    private static final int EF_SEARCH = 10;
    private static final Map<String, ?> HNSW_METHOD_PARAMS = Map.of("ef_search", EF_SEARCH);
    private static final Float MAX_DISTANCE = 1.0f;
    private static final Float MIN_SCORE = 0.5f;
    private static final TermQueryBuilder TERM_QUERY = QueryBuilders.termQuery("field", "value");
    private static final float[] QUERY_VECTOR = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
    protected static final String TEXT_FIELD_NAME = "some_field";
    protected static final String TEXT_VALUE = "some_value";

    @Before
    @Override
    public void setUp() throws Exception {
        super.setUp();
        ClusterSettings clusterSettings = mock(ClusterSettings.class);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
    }

    public void testInvalidK() {
        float[] queryVector = { 1.0f, 1.0f };

        /**
         * -ve k
         */
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder(FIELD_NAME, queryVector, -K));

        /**
         * zero k
         */
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder(FIELD_NAME, queryVector, 0));

        /**
         * k > KNNQueryBuilder.K_MAX
         */
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder(FIELD_NAME, queryVector, KNNQueryBuilder.K_MAX + K));
    }

    public void testInvalidDistance() {
        float[] queryVector = { 1.0f, 1.0f };
        /**
         * null distance
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).maxDistance(null).build()
        );
    }

    public void testInvalidScore() {
        float[] queryVector = { 1.0f, 1.0f };
        /**
         * null min_score
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).minScore(null).build()
        );

        /**
         * negative min_score
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).minScore(-1.0f).build()
        );

        /**
         * min_score = 0
         */
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).minScore(0.0f).build()
        );
    }

    public void testEmptyVector() {
        /**
         * null query vector
         */
        float[] queryVector = null;
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder(FIELD_NAME, queryVector, K));

        /**
         * empty query vector
         */
        float[] queryVector1 = {};
        expectThrows(IllegalArgumentException.class, () -> new KNNQueryBuilder(FIELD_NAME, queryVector1, K));

        /**
         * null query vector with distance
         */
        float[] queryVector2 = null;
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector2).maxDistance(MAX_DISTANCE).build()
        );

        /**
         * empty query vector with distance
         */
        float[] queryVector3 = {};
        expectThrows(
            IllegalArgumentException.class,
            () -> KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector3).maxDistance(MAX_DISTANCE).build()
        );
    }

    @Override
    protected NamedWriteableRegistry writableRegistry() {
        final List<NamedWriteableRegistry.Entry> entries = ClusterModule.getNamedWriteables();
        entries.add(new NamedWriteableRegistry.Entry(QueryBuilder.class, KNNQueryBuilder.NAME, KNNQueryBuilder::new));
        entries.add(new NamedWriteableRegistry.Entry(QueryBuilder.class, TermQueryBuilder.NAME, TermQueryBuilder::new));
        return new NamedWriteableRegistry(entries);
    }

    @SneakyThrows
    public void testDoToQuery_whenNormal_whenDoRadiusSearch_whenDistanceThreshold_thenSucceed() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(MAX_DISTANCE)
            .build();
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        FloatVectorSimilarityQuery query = (FloatVectorSimilarityQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);
        float resultSimilarity = KNNEngine.LUCENE.distanceToRadialThreshold(MAX_DISTANCE, SpaceType.L2);

        assertTrue(query.toString().contains("resultSimilarity=" + resultSimilarity));
        assertTrue(
            query.toString()
                .contains(
                    "traversalSimilarity="
                        + org.opensearch.knn.common.KNNConstants.DEFAULT_LUCENE_RADIAL_SEARCH_TRAVERSAL_SIMILARITY_RATIO * resultSimilarity
                )
        );
    }

    @SneakyThrows
    public void testDoToQuery_whenNormal_whenDoRadiusSearch_whenScoreThreshold_thenSucceed() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(queryVector).minScore(MIN_SCORE).build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        FloatVectorSimilarityQuery query = (FloatVectorSimilarityQuery) knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertTrue(query.toString().contains("resultSimilarity=" + 0.5f));
    }

    @SneakyThrows
    public void testDoToQuery_whenDoRadiusSearch_whenDistanceThreshold_whenFilter_thenSucceed() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(MAX_DISTANCE)
            .filter(TERM_QUERY)
            .build();

        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        Query query = knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertNotNull(query);
        assertTrue(query.getClass().isAssignableFrom(FloatVectorSimilarityQuery.class));
    }

    @SneakyThrows
    public void testDoToQuery_whenDoRadiusSearch_whenScoreThreshold_whenFilter_thenSucceed() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .maxDistance(MAX_DISTANCE)
            .filter(TERM_QUERY)
            .build();
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        MethodComponentContext methodComponentContext = new MethodComponentContext(
            org.opensearch.knn.common.KNNConstants.METHOD_HNSW,
            ImmutableMap.of()
        );
        KNNMethodContext knnMethodContext = new KNNMethodContext(KNNEngine.LUCENE, SpaceType.L2, methodComponentContext);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        Query query = knnQueryBuilder.doToQuery(mockQueryShardContext);
        assertNotNull(query);
        assertTrue(query.getClass().isAssignableFrom(FloatVectorSimilarityQuery.class));
    }

    public void testDoToQuery_ThrowsIllegalArgumentExceptionForUnknownMethodParameter() {

        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.COSINESIMIL,
            new MethodComponentContext("hnsw", Map.of())
        );
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));

        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(K)
            .methodParameters(Map.of("nprobes", 10))
            .build();

        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }

    public void testDoToQuery_InvalidDimensions() {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), 400));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(getDefaultKNNMethodContext(), K));
        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }

    public void testDoToQuery_InvalidFieldType() throws IOException {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder("mynumber", queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        NumberFieldMapper.NumberFieldType mockNumberField = mock(NumberFieldMapper.NumberFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockNumberField);
        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }

    public void testDoToQuery_InvalidZeroFloatVector() {
        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.getSpaceType()).thenReturn(SpaceType.COSINESIMIL);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> knnQueryBuilder.doToQuery(mockQueryShardContext)
        );
        assertEquals(
            String.format(Locale.ROOT, "zero vector is not supported when space type is [%s]", SpaceType.COSINESIMIL.getValue()),
            exception.getMessage()
        );
    }

    public void testDoToQuery_InvalidZeroByteVector() {
        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, queryVector, K);
        Index dummyIndex = new Index("dummy", "dummy");
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.BYTE);
        KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.getSpaceType()).thenReturn(SpaceType.COSINESIMIL);
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);
        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> knnQueryBuilder.doToQuery(mockQueryShardContext)
        );
        assertEquals(
            String.format(Locale.ROOT, "zero vector is not supported when space type is [%s]", SpaceType.COSINESIMIL.getValue()),
            exception.getMessage()
        );
    }

    public void testIgnoreUnmapped() throws IOException {
        float[] queryVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        KNNQueryBuilder.Builder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(K)
            .ignoreUnmapped(true);
        assertTrue(knnQueryBuilder.build().isIgnoreUnmapped());
        Query query = knnQueryBuilder.build().doToQuery(mock(QueryShardContext.class));
        assertNotNull(query);
        assertThat(query, instanceOf(MatchNoDocsQuery.class));
        knnQueryBuilder.ignoreUnmapped(false);
        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.build().doToQuery(mock(QueryShardContext.class)));
    }

    /**
     * Test that radial search (maxDistance/minScore) throws UnsupportedOperationException for engines that don't support it.
     * Currently, only LUCENE supports radial search. JVECTOR does not support radial search.
     *
     * Note: Each engine has different supported methods:
     * - LUCENE supports "hnsw" method
     * - JVECTOR supports "diskann" method only
     *
     * We must use the correct method for each engine to avoid IllegalArgumentException from method validation
     * before reaching the radial search validation code.
     */
    public void testRadialSearch_whenUnsupportedEngine_thenThrowException() {
        List<KNNEngine> unsupportedEngines = Arrays.stream(KNNEngine.values())
            .filter(knnEngine -> !ENGINES_SUPPORTING_RADIAL_SEARCH.contains(knnEngine))
            .collect(Collectors.toList());
        for (KNNEngine knnEngine : unsupportedEngines) {
            // Use engine-specific method to avoid method validation errors before radial search validation
            String methodName = knnEngine == KNNEngine.JVECTOR
                ? org.opensearch.knn.common.KNNConstants.DISK_ANN  // JVECTOR only supports diskann
                : org.opensearch.knn.common.KNNConstants.METHOD_HNSW;  // Other engines support hnsw

            KNNMethodContext knnMethodContext = new KNNMethodContext(
                knnEngine,
                SpaceType.L2,
                new MethodComponentContext(methodName, ImmutableMap.of())
            );

            KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
                .fieldName(FIELD_NAME)
                .vector(QUERY_VECTOR)
                .maxDistance(MAX_DISTANCE)  // This should trigger UnsupportedOperationException for non-LUCENE engines
                .build();

            KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
            QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
            Index dummyIndex = new Index("dummy", "dummy");
            when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
            when(mockQueryShardContext.index()).thenReturn(dummyIndex);
            when(mockKNNVectorField.getVectorDataType()).thenReturn(VectorDataType.FLOAT);
            when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);

            expectThrows(UnsupportedOperationException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
        }
    }

    public void testRadialSearch_whenEfSearchIsSet_whenLuceneEngine_thenThrowException() {
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.L2,
            new MethodComponentContext(org.opensearch.knn.common.KNNConstants.METHOD_HNSW, ImmutableMap.of())
        );

        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .maxDistance(MAX_DISTANCE)
            .methodParameters(Map.of("ef_search", EF_SEARCH))
            .build();

        KNNVectorFieldType mockKNNVectorField = mock(KNNVectorFieldType.class);
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        Index dummyIndex = new Index("dummy", "dummy");
        when(mockKNNVectorField.getKnnMappingConfig()).thenReturn(getMappingConfigForMethodMapping(knnMethodContext, 4));
        when(mockQueryShardContext.index()).thenReturn(dummyIndex);
        when(mockQueryShardContext.fieldMapper(anyString())).thenReturn(mockKNNVectorField);

        expectThrows(IllegalArgumentException.class, () -> knnQueryBuilder.doToQuery(mockQueryShardContext));
    }

    @SneakyThrows
    public void testDoRewrite_whenNoFilter_thenSuccessful() {
        KNNQueryBuilder knnQueryBuilder = new KNNQueryBuilder(FIELD_NAME, QUERY_VECTOR, K);
        QueryBuilder rewritten = knnQueryBuilder.rewrite(mock(QueryRewriteContext.class));
        assertEquals(knnQueryBuilder, rewritten);
    }

    @SneakyThrows
    public void testDoRewrite_whenFilterSet_thenSuccessful() {
        // Given
        QueryBuilder filter = mock(QueryBuilder.class);
        QueryBuilder rewrittenFilter = mock(QueryBuilder.class);
        QueryRewriteContext context = mock(QueryRewriteContext.class);
        when(filter.rewrite(context)).thenReturn(rewrittenFilter);
        KNNQueryBuilder expected = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .filter(rewrittenFilter)
            .k(K)
            .build();

        // When
        KNNQueryBuilder knnQueryBuilder = KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).filter(filter).k(K).build();

        QueryBuilder actual = knnQueryBuilder.rewrite(context);

        assertEquals(knnQueryBuilder, KNNQueryBuilder.builder().fieldName(FIELD_NAME).vector(QUERY_VECTOR).filter(filter).k(K).build());

        // Then
        assertEquals(expected, actual);
    }
}

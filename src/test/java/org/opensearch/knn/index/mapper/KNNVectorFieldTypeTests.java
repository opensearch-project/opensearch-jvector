/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import java.util.Collections;
import java.util.Optional;

import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;
import static org.mockito.Mockito.mock;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.mapper.ArraySourceValueFetcher;
import org.opensearch.index.mapper.ValueFetcher;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.QueryShardException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNVectorIndexFieldData;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.search.lookup.SearchLookup;

public class KNNVectorFieldTypeTests extends KNNTestCase {

    private static final String FIELD_NAME = "test-field";
    private static final int DIMENSION = 3;

    public void testValueFetcher() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, DIMENSION)
        );
        QueryShardContext mockQueryShardContext = mock(QueryShardContext.class);
        ValueFetcher valueFetcher = knnVectorFieldType.valueFetcher(mockQueryShardContext, null, null);
        assertTrue(valueFetcher instanceof ArraySourceValueFetcher);
    }

    public void testTypeName() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, DIMENSION)
        );
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, knnVectorFieldType.typeName());
    }

    public void testExistsQuery() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, DIMENSION)
        );
        QueryShardContext mockContext = mock(QueryShardContext.class);
        Query query = knnVectorFieldType.existsQuery(mockContext);
        assertTrue(query instanceof FieldExistsQuery);
        assertEquals(FIELD_NAME, ((FieldExistsQuery) query).getField());
    }

    public void testTermQuery_throwsException() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, DIMENSION)
        );
        QueryShardContext mockContext = mock(QueryShardContext.class);
        expectThrows(QueryShardException.class, () -> knnVectorFieldType.termQuery(new float[] { 1.0f, 2.0f, 3.0f }, mockContext));
    }

    public void testFielddataBuilder() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, DIMENSION)
        );
        SearchLookup mockSearchLookup = mock(SearchLookup.class);
        IndexFieldData.Builder builder = knnVectorFieldType.fielddataBuilder("test-index", () -> mockSearchLookup);
        assertTrue(builder instanceof KNNVectorIndexFieldData.Builder);
    }

    public void testValueForDisplay_whenFloatVector() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, DIMENSION)
        );
        float[] testVector = new float[] { 1.0f, 2.0f, 3.0f };
        BytesRef serializedVector = new BytesRef(new byte[testVector.length * Float.BYTES]);
        for (int i = 0; i < testVector.length; i++) {
            int bits = Float.floatToIntBits(testVector[i]);
            int offset = i * Float.BYTES;
            serializedVector.bytes[offset] = (byte) (bits & 0xFF);
            serializedVector.bytes[offset + 1] = (byte) ((bits >> 8) & 0xFF);
            serializedVector.bytes[offset + 2] = (byte) ((bits >> 16) & 0xFF);
            serializedVector.bytes[offset + 3] = (byte) ((bits >> 24) & 0xFF);
        }
        serializedVector.length = testVector.length * Float.BYTES;
        Object result = knnVectorFieldType.valueForDisplay(serializedVector);
        assertTrue(result instanceof float[]);
        assertEquals(testVector.length, ((float[]) result).length);
    }

    public void testValueForDisplay_whenByteVector() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.BYTE,
            getMappingConfigForMethodMapping(knnMethodContext, DIMENSION)
        );
        byte[] testVector = new byte[] { 1, 2, 3 };
        BytesRef serializedVector = new BytesRef(testVector);
        Object result = knnVectorFieldType.valueForDisplay(serializedVector);
        assertTrue(result instanceof int[]);
    }

    public void testResolveRescoreContext_whenUserProvidedContext() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, DIMENSION)
        );
        RescoreContext userContext = RescoreContext.builder().oversampleFactor(2.5f).userProvided(true).build();
        RescoreContext result = knnVectorFieldType.resolveRescoreContext(userContext);
        assertSame(userContext, result);
    }

    public void testResolveRescoreContext_whenNullContext() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNMappingConfig mappingConfig = new KNNMappingConfig() {
            @Override
            public int getDimension() {
                return DIMENSION;
            }

            @Override
            public Optional<KNNMethodContext> getKnnMethodContext() {
                return Optional.of(knnMethodContext);
            }

            @Override
            public CompressionLevel getCompressionLevel() {
                return CompressionLevel.x32;
            }

            @Override
            public Mode getMode() {
                return Mode.ON_DISK;
            }
        };
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            mappingConfig
        );
        RescoreContext result = knnVectorFieldType.resolveRescoreContext(null);
        assertNotNull(result);
    }

    public void testTransformQueryVector_whenFloatVector() {
        KNNMethodContext knnMethodContext = getDefaultKNNMethodContext();
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            getMappingConfigForMethodMapping(knnMethodContext, DIMENSION)
        );
        float[] queryVector = new float[] { 3.0f, 4.0f, 0.0f };
        knnVectorFieldType.transformQueryVector(queryVector);
        assertNotNull(queryVector);
        assertEquals(3, queryVector.length);
    }

    public void testTransformQueryVector_whenByteVector() {
        KNNMethodContext knnMethodContext = getDefaultByteKNNMethodContext();
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.BYTE,
            getMappingConfigForMethodMapping(knnMethodContext, DIMENSION)
        );
        float[] queryVector = new float[] { 1.0f, 2.0f, 3.0f };
        float[] originalVector = queryVector.clone();
        knnVectorFieldType.transformQueryVector(queryVector);
        assertArrayEquals(originalVector, queryVector, 0.0001f);
    }

    public void testTransformQueryVector_whenNoMethodContext_throwsException() {
        KNNMappingConfig mappingConfig = getMappingConfigForFlatMapping(DIMENSION);
        KNNVectorFieldType knnVectorFieldType = new KNNVectorFieldType(
            FIELD_NAME,
            Collections.emptyMap(),
            VectorDataType.FLOAT,
            mappingConfig
        );
        float[] queryVector = new float[] { 1.0f, 2.0f, 3.0f };
        expectThrows(IllegalStateException.class, () -> knnVectorFieldType.transformQueryVector(queryVector));
    }
}

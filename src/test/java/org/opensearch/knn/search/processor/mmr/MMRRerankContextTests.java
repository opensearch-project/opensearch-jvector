/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.search.processor.mmr;

import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.search.fetch.subphase.FetchSourceContext;

import java.util.HashMap;
import java.util.Map;

/**
 * Unit tests for MMRRerankContext
 */
public class MMRRerankContextTests extends LuceneTestCase {
    
    // ========== Constructor Tests ==========

    @Test
    public void testDefaultConstructor() {
        MMRRerankContext context = new MMRRerankContext();

        assertNull(context.getOriginalQuerySize());
        assertNull(context.getDiversity());
        assertNull(context.getOriginalFetchSourceContext());
        assertNull(context.getSpaceType());
        assertNull(context.getVectorFieldPath());
        assertNull(context.getVectorDataType());
        assertNull(context.getIndexToVectorFieldPathMap());
    }

    // ========== Getter/Setter Tests ==========

    @Test
    public void testSetAndGetOriginalQuerySize() {
        MMRRerankContext context = new MMRRerankContext();
        
        context.setOriginalQuerySize(10);
        assertEquals(Integer.valueOf(10), context.getOriginalQuerySize());
        
        context.setOriginalQuerySize(100);
        assertEquals(Integer.valueOf(100), context.getOriginalQuerySize());
        
        context.setOriginalQuerySize(null);
        assertNull(context.getOriginalQuerySize());
    }

    @Test
    public void testSetAndGetDiversity() {
        MMRRerankContext context = new MMRRerankContext();
        
        context.setDiversity(0.5f);
        assertEquals(Float.valueOf(0.5f), context.getDiversity());
        
        context.setDiversity(0.0f);
        assertEquals(Float.valueOf(0.0f), context.getDiversity());
        
        context.setDiversity(1.0f);
        assertEquals(Float.valueOf(1.0f), context.getDiversity());
        
        context.setDiversity(null);
        assertNull(context.getDiversity());
    }

    @Test
    public void testSetAndGetOriginalFetchSourceContext() {
        MMRRerankContext context = new MMRRerankContext();
        
        FetchSourceContext fetchSourceContext = FetchSourceContext.FETCH_SOURCE;
        context.setOriginalFetchSourceContext(fetchSourceContext);
        assertEquals(fetchSourceContext, context.getOriginalFetchSourceContext());
        
        context.setOriginalFetchSourceContext(FetchSourceContext.DO_NOT_FETCH_SOURCE);
        assertEquals(FetchSourceContext.DO_NOT_FETCH_SOURCE, context.getOriginalFetchSourceContext());
        
        context.setOriginalFetchSourceContext(null);
        assertNull(context.getOriginalFetchSourceContext());
    }

    @Test
    public void testSetAndGetSpaceType() {
        MMRRerankContext context = new MMRRerankContext();
        
        context.setSpaceType(SpaceType.L2);
        assertEquals(SpaceType.L2, context.getSpaceType());
        
        context.setSpaceType(SpaceType.COSINESIMIL);
        assertEquals(SpaceType.COSINESIMIL, context.getSpaceType());
        
        context.setSpaceType(SpaceType.INNER_PRODUCT);
        assertEquals(SpaceType.INNER_PRODUCT, context.getSpaceType());
        
        context.setSpaceType(null);
        assertNull(context.getSpaceType());
    }

    @Test
    public void testSetAndGetVectorFieldPath() {
        MMRRerankContext context = new MMRRerankContext();
        
        context.setVectorFieldPath("embedding");
        assertEquals("embedding", context.getVectorFieldPath());
        
        context.setVectorFieldPath("user.profile.vector");
        assertEquals("user.profile.vector", context.getVectorFieldPath());
        
        context.setVectorFieldPath(null);
        assertNull(context.getVectorFieldPath());
    }

    @Test
    public void testSetAndGetVectorDataType() {
        MMRRerankContext context = new MMRRerankContext();
        
        context.setVectorDataType(VectorDataType.FLOAT);
        assertEquals(VectorDataType.FLOAT, context.getVectorDataType());
        
        context.setVectorDataType(VectorDataType.BYTE);
        assertEquals(VectorDataType.BYTE, context.getVectorDataType());
        
        context.setVectorDataType(VectorDataType.BINARY);
        assertEquals(VectorDataType.BINARY, context.getVectorDataType());
        
        context.setVectorDataType(null);
        assertNull(context.getVectorDataType());
    }

    @Test
    public void testSetAndGetIndexToVectorFieldPathMap() {
        MMRRerankContext context = new MMRRerankContext();
        
        Map<String, String> map = new HashMap<>();
        map.put("index1", "embedding1");
        map.put("index2", "embedding2");
        
        context.setIndexToVectorFieldPathMap(map);
        assertEquals(map, context.getIndexToVectorFieldPathMap());
        assertEquals("embedding1", context.getIndexToVectorFieldPathMap().get("index1"));
        assertEquals("embedding2", context.getIndexToVectorFieldPathMap().get("index2"));
        
        context.setIndexToVectorFieldPathMap(null);
        assertNull(context.getIndexToVectorFieldPathMap());
    }

    // ========== Integration Tests ==========

    @Test
    public void testCompleteContextSetup() {
        MMRRerankContext context = new MMRRerankContext();
        
        // Set all fields
        context.setOriginalQuerySize(50);
        context.setDiversity(0.7f);
        context.setOriginalFetchSourceContext(FetchSourceContext.FETCH_SOURCE);
        context.setSpaceType(SpaceType.COSINESIMIL);
        context.setVectorFieldPath("document.embedding");
        context.setVectorDataType(VectorDataType.FLOAT);
        
        Map<String, String> indexMap = new HashMap<>();
        indexMap.put("products", "product_vector");
        indexMap.put("reviews", "review_vector");
        context.setIndexToVectorFieldPathMap(indexMap);
        
        // Verify all fields
        assertEquals(Integer.valueOf(50), context.getOriginalQuerySize());
        assertEquals(Float.valueOf(0.7f), context.getDiversity());
        assertEquals(FetchSourceContext.FETCH_SOURCE, context.getOriginalFetchSourceContext());
        assertEquals(SpaceType.COSINESIMIL, context.getSpaceType());
        assertEquals("document.embedding", context.getVectorFieldPath());
        assertEquals(VectorDataType.FLOAT, context.getVectorDataType());
        assertNotNull(context.getIndexToVectorFieldPathMap());
        assertEquals(2, context.getIndexToVectorFieldPathMap().size());
        assertEquals("product_vector", context.getIndexToVectorFieldPathMap().get("products"));
        assertEquals("review_vector", context.getIndexToVectorFieldPathMap().get("reviews"));
    }

    @Test
    public void testPartialContextSetup() {
        MMRRerankContext context = new MMRRerankContext();
        
        // Set only required fields
        context.setOriginalQuerySize(10);
        context.setDiversity(0.5f);
        context.setSpaceType(SpaceType.L2);
        context.setVectorFieldPath("embedding");
        context.setVectorDataType(VectorDataType.FLOAT);
        
        // Verify set fields
        assertEquals(Integer.valueOf(10), context.getOriginalQuerySize());
        assertEquals(Float.valueOf(0.5f), context.getDiversity());
        assertEquals(SpaceType.L2, context.getSpaceType());
        assertEquals("embedding", context.getVectorFieldPath());
        assertEquals(VectorDataType.FLOAT, context.getVectorDataType());
        
        // Verify optional fields remain null
        assertNull(context.getOriginalFetchSourceContext());
        assertNull(context.getIndexToVectorFieldPathMap());
    }

    @Test
    public void testContextModification() {
        MMRRerankContext context = new MMRRerankContext();
        
        // Initial setup
        context.setOriginalQuerySize(10);
        context.setDiversity(0.3f);
        context.setSpaceType(SpaceType.L2);
        
        assertEquals(Integer.valueOf(10), context.getOriginalQuerySize());
        assertEquals(Float.valueOf(0.3f), context.getDiversity());
        assertEquals(SpaceType.L2, context.getSpaceType());
        
        // Modify values
        context.setOriginalQuerySize(20);
        context.setDiversity(0.8f);
        context.setSpaceType(SpaceType.COSINESIMIL);
        
        // Verify modifications
        assertEquals(Integer.valueOf(20), context.getOriginalQuerySize());
        assertEquals(Float.valueOf(0.8f), context.getDiversity());
        assertEquals(SpaceType.COSINESIMIL, context.getSpaceType());
    }

    @Test
    public void testEmptyIndexToVectorFieldPathMap() {
        MMRRerankContext context = new MMRRerankContext();
        
        Map<String, String> emptyMap = new HashMap<>();
        context.setIndexToVectorFieldPathMap(emptyMap);
        
        assertNotNull(context.getIndexToVectorFieldPathMap());
        assertTrue(context.getIndexToVectorFieldPathMap().isEmpty());
        assertEquals(0, context.getIndexToVectorFieldPathMap().size());
    }

    @Test
    public void testMultipleIndexMappings() {
        MMRRerankContext context = new MMRRerankContext();
        
        Map<String, String> indexMap = new HashMap<>();
        indexMap.put("index-1", "field1");
        indexMap.put("index-2", "field2");
        indexMap.put("index-3", "field3");
        indexMap.put("index-4", "field4");
        indexMap.put("index-5", "field5");
        
        context.setIndexToVectorFieldPathMap(indexMap);
        
        assertEquals(5, context.getIndexToVectorFieldPathMap().size());
        assertEquals("field1", context.getIndexToVectorFieldPathMap().get("index-1"));
        assertEquals("field5", context.getIndexToVectorFieldPathMap().get("index-5"));
    }

    @Test
    public void testDiversityBoundaryValues() {
        MMRRerankContext context = new MMRRerankContext();
        
        // Test minimum diversity (pure relevance)
        context.setDiversity(0.0f);
        assertEquals(Float.valueOf(0.0f), context.getDiversity());
        
        // Test maximum diversity (pure diversity)
        context.setDiversity(1.0f);
        assertEquals(Float.valueOf(1.0f), context.getDiversity());
        
        // Test mid-range diversity
        context.setDiversity(0.5f);
        assertEquals(Float.valueOf(0.5f), context.getDiversity());
    }

    @Test
    public void testQuerySizeVariations() {
        MMRRerankContext context = new MMRRerankContext();
        
        // Small query size
        context.setOriginalQuerySize(1);
        assertEquals(Integer.valueOf(1), context.getOriginalQuerySize());
        
        // Medium query size
        context.setOriginalQuerySize(50);
        assertEquals(Integer.valueOf(50), context.getOriginalQuerySize());
        
        // Large query size
        context.setOriginalQuerySize(1000);
        assertEquals(Integer.valueOf(1000), context.getOriginalQuerySize());
    }

    @Test
    public void testAllSpaceTypes() {
        MMRRerankContext context = new MMRRerankContext();
        
        SpaceType[] spaceTypes = { SpaceType.L2, SpaceType.COSINESIMIL, SpaceType.INNER_PRODUCT };
        
        for (SpaceType spaceType : spaceTypes) {
            context.setSpaceType(spaceType);
            assertEquals(spaceType, context.getSpaceType());
        }
    }

    @Test
    public void testAllVectorDataTypes() {
        MMRRerankContext context = new MMRRerankContext();
        
        VectorDataType[] dataTypes = { VectorDataType.FLOAT, VectorDataType.BYTE, VectorDataType.BINARY };
        
        for (VectorDataType dataType : dataTypes) {
            context.setVectorDataType(dataType);
            assertEquals(dataType, context.getVectorDataType());
        }
    }

    @Test
    public void testNestedVectorFieldPath() {
        MMRRerankContext context = new MMRRerankContext();
        
        String[] paths = {
            "embedding",
            "user.embedding",
            "document.metadata.vector",
            "product.details.features.embedding"
        };
        
        for (String path : paths) {
            context.setVectorFieldPath(path);
            assertEquals(path, context.getVectorFieldPath());
        }
    }
}
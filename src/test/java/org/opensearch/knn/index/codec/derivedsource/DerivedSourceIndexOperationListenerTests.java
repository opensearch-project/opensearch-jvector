/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.Term;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.mapper.ParseContext;
import org.opensearch.index.mapper.ParsedDocument;
import org.opensearch.index.mapper.SourceFieldMapper;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.DerivedKnnFloatVectorField;
import org.opensearch.knn.index.codec.KNN10010Codec.KNN10010DerivedSourceStoredFieldsWriter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class DerivedSourceIndexOperationListenerTests extends KNNTestCase {

    public void testPreIndexWithMultiLevelNestedDocuments() throws Exception {
        // Create multi-level nested structure similar to testNestedField test:
        // nested_1 with test_vector
        // nested_2 with test_vector and nested_3 with test_vector
        
        // Original source structure
        Map<String, Object> originalSourceMap = Map.of(
            "nested_1", List.of(
                Map.of("name", "n1_obj1", "test_vector", new int[]{1, 2, 3}),
                Map.of("name", "n1_obj2", "test_vector", new int[]{4, 5, 6})
            ),
            "nested_2", List.of(
                Map.of(
                    "name", "n2_obj1",
                    "test_vector", new int[]{7, 8, 9},
                    "nested_3", List.of(
                        Map.of("name", "n3_obj1", "test_vector", new int[]{10, 11, 12}),
                        Map.of("name", "n3_obj2", "test_vector", new int[]{13, 14, 15})
                    )
                )
            )
        );
        
        BytesStreamOutput bStream = new BytesStreamOutput();
        XContentBuilder builder = MediaTypeRegistry.contentBuilder(XContentType.JSON, bStream).map(originalSourceMap);
        builder.close();
        BytesReference originalSource = bStream.bytes();

        // Backend vectors for each nested document
        // Lucene flattens: [n1_child1, n1_child2, n3_child1, n3_child2, n2_child1, parent]
        float[] n1_vec1 = {1.0f, 2.0f, 3.0f};
        float[] n1_vec2 = {4.0f, 5.0f, 6.0f};
        float[] n2_vec1 = {7.0f, 8.0f, 9.0f};
        float[] n3_vec1 = {10.0f, 11.0f, 12.0f};
        float[] n3_vec2 = {13.0f, 14.0f, 15.0f};
        
        // Create child documents in Lucene's flattened order
        ParseContext.Document n1_child1 = new ParseContext.Document();
        n1_child1.add(new DerivedKnnFloatVectorField("nested_1.test_vector", n1_vec1, true));
        
        ParseContext.Document n1_child2 = new ParseContext.Document();
        n1_child2.add(new DerivedKnnFloatVectorField("nested_1.test_vector", n1_vec2, true));
        
        ParseContext.Document n3_child1 = new ParseContext.Document();
        n3_child1.add(new DerivedKnnFloatVectorField("nested_2.nested_3.test_vector", n3_vec1, true));
        
        ParseContext.Document n3_child2 = new ParseContext.Document();
        n3_child2.add(new DerivedKnnFloatVectorField("nested_2.nested_3.test_vector", n3_vec2, true));
        
        ParseContext.Document n2_child1 = new ParseContext.Document();
        n2_child1.add(new DerivedKnnFloatVectorField("nested_2.test_vector", n2_vec1, true));
        
        // Create parent document with source
        ParseContext.Document parentDoc = new ParseContext.Document();
        parentDoc.add(new StoredField(SourceFieldMapper.NAME, originalSource.toBytesRef()));

        Engine.Index operation = new Engine.Index(
            new Term("test-id"),
            1,
            new ParsedDocument(null, null, null, null, List.of(n1_child1, n1_child2, n3_child1, n3_child2, n2_child1, parentDoc), originalSource, XContentType.JSON, null)
        );

        DerivedSourceIndexOperationListener derivedSourceIndexOperationListener = new DerivedSourceIndexOperationListener();
        operation = derivedSourceIndexOperationListener.preIndex(null, operation);
        
        // Check translog source (should have actual vectors injected)
        Tuple<? extends MediaType, Map<String, Object>> translogSource = XContentHelper.convertToMap(
            operation.parsedDoc().source(),
            true,
            operation.parsedDoc().getMediaType()
        );
        
        // Verify nested_1 vectors in translog
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> nested1InTranslog = (List<Map<String, Object>>) translogSource.v2().get("nested_1");
        assertNotNull("nested_1 should exist in translog", nested1InTranslog);
        assertEquals("nested_1 should have 2 objects", 2, nested1InTranslog.size());
        assertEquals("nested_1[0] vector should match", List.of(1.0, 2.0, 3.0), nested1InTranslog.get(0).get("test_vector"));
        assertEquals("nested_1[1] vector should match", List.of(4.0, 5.0, 6.0), nested1InTranslog.get(1).get("test_vector"));
        
        // Verify nested_2 vectors in translog
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> nested2InTranslog = (List<Map<String, Object>>) translogSource.v2().get("nested_2");
        assertNotNull("nested_2 should exist in translog", nested2InTranslog);
        assertEquals("nested_2 should have 1 object", 1, nested2InTranslog.size());
        assertEquals("nested_2[0] vector should match", List.of(7.0, 8.0, 9.0), nested2InTranslog.get(0).get("test_vector"));
        
        // Verify nested_3 vectors in translog
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> nested3InTranslog = (List<Map<String, Object>>) nested2InTranslog.get(0).get("nested_3");
        assertNotNull("nested_3 should exist in translog", nested3InTranslog);
        assertEquals("nested_3 should have 2 objects", 2, nested3InTranslog.size());
        assertEquals("nested_3[0] vector should match", List.of(10.0, 11.0, 12.0), nested3InTranslog.get(0).get("test_vector"));
        assertEquals("nested_3[1] vector should match", List.of(13.0, 14.0, 15.0), nested3InTranslog.get(1).get("test_vector"));
        
        // Check stored field source (should have masks)
        IndexableField field = operation.parsedDoc().rootDoc().getField(SourceFieldMapper.CONTENT_TYPE);
        assertTrue(field instanceof StoredField);
        StoredField sourceField = (StoredField) field;
        BytesRef bytesRef = sourceField.binaryValue();
        Tuple<? extends MediaType, Map<String, Object>> maskedSource = XContentHelper.convertToMap(
            new BytesArray(bytesRef.bytes, bytesRef.offset, bytesRef.length),
            true,
            operation.parsedDoc().getMediaType()
        );
        
        // Verify vectors are masked in stored source
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> nested1InStored = (List<Map<String, Object>>) maskedSource.v2().get("nested_1");
        assertEquals("nested_1[0] vector should be masked", KNN10010DerivedSourceStoredFieldsWriter.MASK.intValue(), nested1InStored.get(0).get("test_vector"));
        assertEquals("nested_1[1] vector should be masked", KNN10010DerivedSourceStoredFieldsWriter.MASK.intValue(), nested1InStored.get(1).get("test_vector"));
        
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> nested2InStored = (List<Map<String, Object>>) maskedSource.v2().get("nested_2");
        assertEquals("nested_2[0] vector should be masked", KNN10010DerivedSourceStoredFieldsWriter.MASK.intValue(), nested2InStored.get(0).get("test_vector"));
        
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> nested3InStored = (List<Map<String, Object>>) nested2InStored.get(0).get("nested_3");
        assertEquals("nested_3[0] vector should be masked", KNN10010DerivedSourceStoredFieldsWriter.MASK.intValue(), nested3InStored.get(0).get("test_vector"));
        assertEquals("nested_3[1] vector should be masked", KNN10010DerivedSourceStoredFieldsWriter.MASK.intValue(), nested3InStored.get(1).get("test_vector"));
    }

    public void testPreIndex() throws Exception {
        String fieldName = "test-vector";
        int[] userVector = { 1, 2, 3, 4 };
        float[] backendVector = { 1.0f, 2.0f, 3.0f, 4.0f };
        List<Double> expectedOutputAsList = new ArrayList<>(List.of(1.0, 2.0, 3.0, 4.0));

        Map<String, Object> originalSourceMap = Map.of(fieldName, userVector);
        BytesStreamOutput bStream = new BytesStreamOutput();
        XContentBuilder builder = MediaTypeRegistry.contentBuilder(XContentType.JSON, bStream).map(originalSourceMap);
        builder.close();
        BytesReference originalSource = bStream.bytes();

        ParseContext.Document document = new ParseContext.Document();
        document.add(new DerivedKnnFloatVectorField(fieldName, backendVector, true));
        document.add(new StoredField(SourceFieldMapper.NAME, originalSource.toBytesRef()));

        Engine.Index operation = new Engine.Index(
            new Term("test-iud"),
            1,
            new ParsedDocument(null, null, null, null, List.of(document), originalSource, XContentType.JSON, null)
        );

        DerivedSourceIndexOperationListener derivedSourceIndexOperationListener = new DerivedSourceIndexOperationListener();
        operation = derivedSourceIndexOperationListener.preIndex(null, operation);
        Tuple<? extends MediaType, Map<String, Object>> modifiedSource = XContentHelper.convertToMap(
            operation.parsedDoc().source(),
            true,
            operation.parsedDoc().getMediaType()
        );

        assertEquals(expectedOutputAsList, modifiedSource.v2().get(fieldName));
        IndexableField field = operation.parsedDoc().rootDoc().getField(SourceFieldMapper.CONTENT_TYPE);
        assertTrue(field instanceof StoredField);
        StoredField sourceField = (StoredField) field;
        assertEquals(sourceField.binaryValue(), sourceField.storedValue().getBinaryValue());
        BytesRef bytesRef = sourceField.binaryValue();
        Tuple<? extends MediaType, Map<String, Object>> maskedSourceBinaryValueMap = XContentHelper.convertToMap(
            new BytesArray(bytesRef.bytes, bytesRef.offset, bytesRef.length),
            true,
            operation.parsedDoc().getMediaType()
        );
        assertEquals(KNN10010DerivedSourceStoredFieldsWriter.MASK.intValue(), maskedSourceBinaryValueMap.v2().get(fieldName));
    }
}

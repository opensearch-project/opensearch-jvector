/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

public class MethodComponentContextTests extends KNNTestCase {

    private static final String TEST_METHOD = "test-method";
    private static final String ENCODER = "encoder";

    public void testConstructor() {
        String name = "test-component";
        Map<String, Object> parameters = Map.of("param1", 10, "param2", "value");

        MethodComponentContext context = new MethodComponentContext(name, parameters);

        assertEquals(name, context.getName());
        assertEquals(parameters, context.getParameters());
    }

    public void testCopyConstructor_withSimpleParameters() {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("param1", 10);
        parameters.put("param2", true);

        MethodComponentContext original = new MethodComponentContext("test-component", parameters);
        MethodComponentContext copy = new MethodComponentContext(original);

        assertEquals(original.getName(), copy.getName());
        assertEquals(original.getParameters(), copy.getParameters());

        // Verify deep copy
        copy.getParameters().put("param3", 20);
        assertFalse(original.getParameters().containsKey("param3"));
    }

    public void testCopyConstructor_withNestedContext() {
        MethodComponentContext nestedContext = new MethodComponentContext("nested", Map.of("nested_param", 5));
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("param1", 10);
        parameters.put(ENCODER, nestedContext);

        MethodComponentContext original = new MethodComponentContext("parent", parameters);
        MethodComponentContext copy = new MethodComponentContext(original);

        MethodComponentContext copiedNested = (MethodComponentContext) copy.getParameters().get(ENCODER);
        MethodComponentContext originalNested = (MethodComponentContext) original.getParameters().get(ENCODER);

        assertNotSame(originalNested, copiedNested);
        assertEquals(originalNested.getName(), copiedNested.getName());
    }

    public void testCopyConstructor_withNull() {
        MethodComponentContext nullContext = null;
        expectThrows(IllegalArgumentException.class, () -> new MethodComponentContext(nullContext));
    }

    public void testParse_withValidInput() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, TEST_METHOD)
            .startObject(PARAMETERS)
            .field("param1", 10)
            .field("param2", true)
            .endObject()
            .endObject();

        Map<String, Object> in = xContentBuilderToMap(builder);
        MethodComponentContext context = MethodComponentContext.parse(in);

        assertEquals(TEST_METHOD, context.getName());
        assertEquals(10, context.getParameters().get("param1"));
        assertEquals(true, context.getParameters().get("param2"));
    }

    public void testParse_withNestedContext() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, "parent")
            .startObject(PARAMETERS)
            .field("param1", 10)
            .startObject(ENCODER)
            .field(NAME, "nested")
            .startObject(PARAMETERS)
            .field("nested_param", 5)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Map<String, Object> in = xContentBuilderToMap(builder);
        MethodComponentContext context = MethodComponentContext.parse(in);

        MethodComponentContext encoderContext = (MethodComponentContext) context.getParameters().get(ENCODER);
        assertEquals("nested", encoderContext.getName());
        assertEquals(5, encoderContext.getParameters().get("nested_param"));
    }

    public void testParse_withNullParameters() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, TEST_METHOD)
            .field(PARAMETERS, (String) null)
            .endObject();

        Map<String, Object> in = xContentBuilderToMap(builder);
        MethodComponentContext context = MethodComponentContext.parse(in);

        assertTrue(context.getParameters().isEmpty());
    }

    public void testParse_withMissingName() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PARAMETERS)
            .field("param1", 10)
            .endObject()
            .endObject();

        Map<String, Object> in = xContentBuilderToMap(builder);
        expectThrows(MapperParsingException.class, () -> MethodComponentContext.parse(in));
    }

    public void testParse_withInvalidInputType() {
        expectThrows(MapperParsingException.class, () -> MethodComponentContext.parse("invalid"));
    }

    public void testParse_withInvalidNameType() {
        Map<String, Object> invalidName = Map.of(NAME, 123, PARAMETERS, Map.of());
        expectThrows(MapperParsingException.class, () -> MethodComponentContext.parse(invalidName));
    }

    public void testParse_withInvalidParametersType() {
        Map<String, Object> invalidParams = Map.of(NAME, TEST_METHOD, PARAMETERS, "invalid");
        expectThrows(MapperParsingException.class, () -> MethodComponentContext.parse(invalidParams));
    }

    public void testParse_withInvalidParameterKey() {
        Map<String, Object> invalidKey = Map.of(NAME, TEST_METHOD, "invalid_key", "value");
        expectThrows(MapperParsingException.class, () -> MethodComponentContext.parse(invalidKey));
    }

    public void testToXContent_withSimpleParameters() throws IOException {
        MethodComponentContext context = new MethodComponentContext(TEST_METHOD, Map.of("param1", 10, "param2", true));

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        context.toXContent(builder, null);
        builder.endObject();

        Map<String, Object> result = xContentBuilderToMap(builder);
        assertEquals(TEST_METHOD, result.get(NAME));

        @SuppressWarnings("unchecked")
        Map<String, Object> resultParams = (Map<String, Object>) result.get(PARAMETERS);
        assertEquals(10, resultParams.get("param1"));
        assertEquals(true, resultParams.get("param2"));
    }

    public void testToXContent_withNestedContext() throws IOException {
        MethodComponentContext nestedContext = new MethodComponentContext("nested", Map.of("nested_param", 5));
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("param1", 10);
        parameters.put(ENCODER, nestedContext);

        MethodComponentContext context = new MethodComponentContext("parent", parameters);
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        context.toXContent(builder, null);
        builder.endObject();

        Map<String, Object> result = xContentBuilderToMap(builder);

        @SuppressWarnings("unchecked")
        Map<String, Object> encoderMap = (Map<String, Object>) ((Map<String, Object>) result.get(PARAMETERS)).get(ENCODER);
        assertEquals("nested", encoderMap.get(NAME));
    }

    public void testToXContent_withNullParameters() throws IOException {
        MethodComponentContext context = new MethodComponentContext(TEST_METHOD, null);
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        context.toXContent(builder, null);
        builder.endObject();

        Map<String, Object> result = xContentBuilderToMap(builder);
        assertNull(result.get(PARAMETERS));
    }

    public void testSerialization_withSimpleParameters() throws IOException {
        MethodComponentContext original = new MethodComponentContext(TEST_METHOD, Map.of("param1", 10, "param2", true));

        BytesStreamOutput output = new BytesStreamOutput();
        original.writeTo(output);

        StreamInput input = output.bytes().streamInput();
        MethodComponentContext deserialized = new MethodComponentContext(input);

        assertEquals(original.getName(), deserialized.getName());
        assertEquals(original.getParameters(), deserialized.getParameters());
    }

    public void testSerialization_withNestedContext() throws IOException {
        MethodComponentContext nestedContext = new MethodComponentContext("nested", Map.of("nested_param", 5));
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("param1", 10);
        parameters.put(ENCODER, nestedContext);

        MethodComponentContext original = new MethodComponentContext("parent", parameters);

        BytesStreamOutput output = new BytesStreamOutput();
        original.writeTo(output);

        StreamInput input = output.bytes().streamInput();
        MethodComponentContext deserialized = new MethodComponentContext(input);

        assertEquals(original.getName(), deserialized.getName());

        MethodComponentContext deserializedNested = (MethodComponentContext) deserialized.getParameters().get(ENCODER);
        assertEquals("nested", deserializedNested.getName());
        assertEquals(5, deserializedNested.getParameters().get("nested_param"));
    }

    public void testEquals() {
        MethodComponentContext context1 = new MethodComponentContext(TEST_METHOD, Map.of("param1", 10));
        MethodComponentContext context2 = new MethodComponentContext(TEST_METHOD, Map.of("param1", 10));
        MethodComponentContext context3 = new MethodComponentContext(TEST_METHOD, Map.of("param1", 20));
        MethodComponentContext context4 = new MethodComponentContext("different", Map.of("param1", 10));

        assertEquals(context1, context2);
        assertNotEquals(context1, context3);
        assertNotEquals(context1, context4);
        assertNotEquals(context1, null);
        assertNotEquals(context1, "string");
    }

    public void testHashCode() {
        MethodComponentContext context1 = new MethodComponentContext(TEST_METHOD, Map.of("param1", 10));
        MethodComponentContext context2 = new MethodComponentContext(TEST_METHOD, Map.of("param1", 10));

        assertEquals(context1.hashCode(), context2.hashCode());
    }

    public void testFromClusterStateString_withSimpleParameters() {
        MethodComponentContext context = MethodComponentContext.fromClusterStateString("{name=test;parameters=[param1=10;param2=5;]}");

        assertEquals("test", context.getName());
        assertEquals(10, context.getParameters().get("param1"));
        assertEquals(5, context.getParameters().get("param2"));
    }

    public void testFromClusterStateString_withNestedContext() {
        MethodComponentContext context = MethodComponentContext.fromClusterStateString(
            "{name=parent;parameters=[param1=10;encoder={name=nested;parameters=[nested_param=5;]};]}"
        );

        assertEquals("parent", context.getName());

        MethodComponentContext nestedContext = (MethodComponentContext) context.getParameters().get(ENCODER);
        assertEquals("nested", nestedContext.getName());
        assertEquals(5, nestedContext.getParameters().get("nested_param"));
    }

    public void testFromClusterStateString_withBooleanParameters() {
        MethodComponentContext context = MethodComponentContext.fromClusterStateString(
            "{name=test;parameters=[param1=true;param2=false;]}"
        );

        assertEquals(true, context.getParameters().get("param1"));
        assertEquals(false, context.getParameters().get("param2"));
    }

    public void testFromClusterStateString_withInvalidFormat() {
        expectThrows(IllegalArgumentException.class, () -> MethodComponentContext.fromClusterStateString("invalid"));
    }

    public void testFromClusterStateString_withMissingClosingBrace() {
        expectThrows(
            IllegalArgumentException.class,
            () -> MethodComponentContext.fromClusterStateString("{name=test;parameters=[param1=10;]")
        );
    }

    public void testFromClusterStateString_withInvalidParameterValue() {
        expectThrows(
            IllegalArgumentException.class,
            () -> MethodComponentContext.fromClusterStateString("{name=test;parameters=[param1=invalid_value;]}")
        );
    }

    public void testRoundTrip_XContent() throws IOException {
        MethodComponentContext nestedContext = new MethodComponentContext("nested", Map.of("nested_param", 5));
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("param1", 10);
        parameters.put("param2", true);
        parameters.put(ENCODER, nestedContext);

        MethodComponentContext original = new MethodComponentContext(TEST_METHOD, parameters);

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        original.toXContent(builder, null);
        builder.endObject();

        Map<String, Object> map = xContentBuilderToMap(builder);
        MethodComponentContext parsedContext = MethodComponentContext.parse(map);

        assertEquals(original, parsedContext);
    }

    public void testRoundTrip_Serialization() throws IOException {
        MethodComponentContext nestedContext = new MethodComponentContext("nested", Map.of("nested_param", 5));
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("param1", 10);
        parameters.put("param2", true);
        parameters.put(ENCODER, nestedContext);

        MethodComponentContext original = new MethodComponentContext(TEST_METHOD, parameters);

        BytesStreamOutput output = new BytesStreamOutput();
        original.writeTo(output);

        StreamInput input = output.bytes().streamInput();
        MethodComponentContext deserializedContext = new MethodComponentContext(input);

        assertEquals(original, deserializedContext);
    }
}

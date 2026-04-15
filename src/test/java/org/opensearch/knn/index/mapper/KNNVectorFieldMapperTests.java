/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.opensearch.Version;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.ValidationException;
import org.opensearch.common.settings.IndexScopedSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.ContentPath;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.index.mapper.MapperParsingException;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.Version.CURRENT;
import static org.opensearch.knn.DerivedSourceUtils.TEST_DIMENSION;
import static org.opensearch.knn.common.KNNConstants.*;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;
import static org.opensearch.knn.index.VectorDataType.SUPPORTED_VECTOR_DATA_TYPES;
import static org.opensearch.knn.index.engine.CommonTestUtils.KNN_VECTOR_TYPE;
import static org.opensearch.knn.index.engine.CommonTestUtils.TYPE_FIELD_NAME;

@Log4j2
public class KNNVectorFieldMapperTests extends KNNTestCase {
    private static final String DIMENSION_FIELD_NAME = "dimension";
    private static final String TEST_INDEX_NAME = "test-index-name";
    private static final String TEST_FIELD_NAME = "test-field-name";

    public void testBuilder_getParameters() {
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder(TEST_FIELD_NAME, CURRENT, null, null);

        assertEquals(9, builder.getParameters().size());
        List<String> actualParams = builder.getParameters().stream().map(a -> a.name).collect(Collectors.toList());
        List<String> expectedParams = Arrays.asList(
            "store",
            "doc_values",
            DIMENSION,
            VECTOR_DATA_TYPE_FIELD,
            "meta",
            KNN_METHOD,
            MODE_PARAMETER,
            COMPRESSION_LEVEL_PARAMETER,
            KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE
        );
        assertEquals(expectedParams, actualParams);
    }

    public void testBuilder_build() {
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder(TEST_FIELD_NAME, CURRENT, null, null);
        builder.setOriginalParameters(new OriginalMappingParameters(builder));
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);

        assertTrue(knnVectorFieldMapper instanceof FlatVectorFieldMapper);
    }

    public void testTypeParser_build_fromKnnMethodContext() throws IOException {
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();
        SpaceType spaceType = SpaceType.L2;
        int mParameter = 17;

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, TEST_DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, DISK_ANN)
            .field(KNN_ENGINE, KNNEngine.JVECTOR.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mParameter)
            .endObject()
            .endObject()
            .endObject();

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            TEST_FIELD_NAME,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(TEST_INDEX_NAME, settings)
        );

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);

        assertTrue(knnVectorFieldMapper instanceof LuceneFieldMapper);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(spaceType, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getSpaceType());
        assertEquals(
            mParameter,
            knnVectorFieldMapper.fieldType()
                .getKnnMappingConfig()
                .getKnnMethodContext()
                .get()
                .getMethodComponentContext()
                .getParameters()
                .get(METHOD_PARAMETER_M)
        );
    }

    public void testTypeParser_parse_fromKnnMethodContext_invalidDimension() throws IOException {
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();
        SpaceType spaceType = SpaceType.L2;
        int mParameter = 17;

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 20000)
            .startObject(KNN_METHOD)
            .field(NAME, DISK_ANN)
            .field(KNN_ENGINE, KNNEngine.JVECTOR.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mParameter)
            .endObject()
            .endObject()
            .endObject();

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        IllegalArgumentException ex = expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(TEST_FIELD_NAME, xContentBuilderToMap(xContentBuilder), buildParserContext(TEST_INDEX_NAME, settings))
        );
        assertTrue(
            ex.getMessage().contains("Validation Failed: 1: Dimension value cannot be greater than 16000 for vector with engine: jvector")
        );
    }

    @SneakyThrows
    public void testTypeParser_parse_invalidVectorDataType() {
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();
        SpaceType spaceType = SpaceType.L2;
        int mParameter = 17;
        String vectorDataType = "invalid";

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, TEST_DIMENSION)
            .field(VECTOR_DATA_TYPE_FIELD, vectorDataType)
            .startObject(KNN_METHOD)
            .field(NAME, DISK_ANN)
            .field(KNN_ENGINE, KNNEngine.JVECTOR.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mParameter)
            .endObject()
            .endObject()
            .endObject();

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        IllegalArgumentException ex = expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(TEST_FIELD_NAME, xContentBuilderToMap(xContentBuilder), buildParserContext(TEST_INDEX_NAME, settings))
        );

        assertEquals(
            String.format(
                Locale.ROOT,
                "Invalid value provided for [%s] field. Supported values are [%s]",
                VECTOR_DATA_TYPE_FIELD,
                SUPPORTED_VECTOR_DATA_TYPES
            ),
            ex.getMessage()
        );
    }

    public void testTypeParser_parse_fromKnnMethodContext_invalidSpaceType() throws IOException {
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();
        int mParameter = 17;

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, TEST_DIMENSION)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L1.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, mParameter)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, 100)
            .endObject()
            .endObject()
            .endObject();

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        expectThrows(
            ValidationException.class,
            () -> typeParser.parse(TEST_FIELD_NAME, xContentBuilderToMap(xContentBuilder), buildParserContext(TEST_INDEX_NAME, settings))
        );
    }

    public void testKNNVectorFieldMapperMerge_whenModeAndCompressionIsPresent_thenSuccess() throws IOException {
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        int dimension = 133;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
            .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            TEST_FIELD_NAME,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(TEST_INDEX_NAME, settings)
        );
        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper1 = builder.build(builderContext);

        // merge with itself - should be successful
        KNNVectorFieldMapper knnVectorFieldMapperMerge1 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper1);
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getKnnMethodContext().get(),
            knnVectorFieldMapperMerge1.fieldType().getKnnMappingConfig().getKnnMethodContext().get()
        );

        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getCompressionLevel(),
            knnVectorFieldMapperMerge1.fieldType().getKnnMappingConfig().getCompressionLevel()
        );
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getMode(),
            knnVectorFieldMapperMerge1.fieldType().getKnnMappingConfig().getMode()
        );

        // merge with another mapper of the same field with same context
        KNNVectorFieldMapper knnVectorFieldMapper2 = builder.build(builderContext);
        KNNVectorFieldMapper knnVectorFieldMapperMerge2 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper2);
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getKnnMethodContext().get(),
            knnVectorFieldMapperMerge2.fieldType().getKnnMappingConfig().getKnnMethodContext().get()
        );

        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getCompressionLevel(),
            knnVectorFieldMapperMerge2.fieldType().getKnnMappingConfig().getCompressionLevel()
        );
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getMode(),
            knnVectorFieldMapperMerge2.fieldType().getKnnMappingConfig().getMode()
        );
    }

    public void testKNNVectorFieldMapper_merge_fromKnnMethodContext() throws IOException {
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        int dimension = 133;
        int efConstruction = 321;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, DISK_ANN)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
            .endObject()
            .endObject()
            .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            TEST_FIELD_NAME,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(TEST_INDEX_NAME, settings)
        );

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper1 = builder.build(builderContext);

        // merge with itself - should be successful
        KNNVectorFieldMapper knnVectorFieldMapperMerge1 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper1);
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getKnnMethodContext().get(),
            knnVectorFieldMapperMerge1.fieldType().getKnnMappingConfig().getKnnMethodContext().get()
        );

        // merge with another mapper of the same field with same context
        KNNVectorFieldMapper knnVectorFieldMapper2 = builder.build(builderContext);
        KNNVectorFieldMapper knnVectorFieldMapperMerge2 = (KNNVectorFieldMapper) knnVectorFieldMapper1.merge(knnVectorFieldMapper2);
        assertEquals(
            knnVectorFieldMapper1.fieldType().getKnnMappingConfig().getKnnMethodContext().get(),
            knnVectorFieldMapperMerge2.fieldType().getKnnMappingConfig().getKnnMethodContext().get()
        );

        // merge with another mapper of the same field with different context
        xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, DISK_ANN)
            .endObject()
            .endObject();

        builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            TEST_FIELD_NAME,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(TEST_INDEX_NAME, settings)
        );
        KNNVectorFieldMapper knnVectorFieldMapper3 = builder.build(builderContext);
        expectThrows(IllegalArgumentException.class, () -> knnVectorFieldMapper1.merge(knnVectorFieldMapper3));
    }

    public void testKNNVectorFieldMapper_UpdateDimensionParameter_Succeeds() throws IOException {
        String fieldName = TEST_FIELD_NAME;
        String indexName = TEST_INDEX_NAME;

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        // Define updated mapping with the same method parameter
        XContentBuilder updatedMapping = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 8)
            .startObject(KNN_METHOD)
            .field(NAME, DISK_ANN)
            .field(KNN_ENGINE, KNNEngine.JVECTOR.getName())
            .endObject()
            .endObject();

        KNNVectorFieldMapper.Builder updatedBuilder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(updatedMapping),
            buildParserContext(indexName, settings)
        );

        assertEquals(8, updatedBuilder.getOriginalParameters().getDimension());
    }

    public void testKNNVectorFieldMapper_PartialUpdateMethodParameter_ThrowsException() throws IOException {
        String fieldName = TEST_FIELD_NAME;
        String indexName = TEST_INDEX_NAME;

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        XContentBuilder updatedMapping = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 4)
            .startObject(KNN_METHOD)
            .field(NAME, "")
            .field(KNN_ENGINE, KNNEngine.JVECTOR.getName())
            .endObject()
            .endObject();

        MapperParsingException exception = expectThrows(
            MapperParsingException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(updatedMapping), buildParserContext(indexName, settings))
        );

        assertTrue(exception.getMessage().contains("name needs to be set"));
    }

    public void testTypeParser_withSpaceTypeAndMode_thenSuccess() throws IOException {
        // Check that knnMethodContext takes precedent over both model and legacy
        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        SpaceType topLevelSpaceType = SpaceType.INNER_PRODUCT;
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION, TEST_DIMENSION)
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x16.getName())
            .field(KNNConstants.TOP_LEVEL_PARAMETER_SPACE_TYPE, topLevelSpaceType.getValue())
            .endObject();
        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildParserContext("test", settings)
        );

        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(topLevelSpaceType, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getSpaceType());
    }

    public void testSpaceType_build_fromLegacy() throws IOException {
        // Check legacy is picked up if method context are not set
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 12)
            .endObject();

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildLegacyParserContext("test", settings, Version.V_2_15_0)
        );

        // Setup settings
        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(SpaceType.L2, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getSpaceType());
    }

    public void testBuilder_build_fromLegacy() throws IOException {
        // Check legacy is picked up if method context are not set
        int m = 17;
        int efConstruction = 17;
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 12)
            .endObject();

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            "test-field-name-1",
            xContentBuilderToMap(xContentBuilder),
            buildParserContext("test", settings)
        );

        // Setup settings
        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().isPresent());
        assertEquals(SpaceType.L2, knnVectorFieldMapper.fieldType().getKnnMappingConfig().getKnnMethodContext().get().getSpaceType());
    }

    public void testBuilder_parse_fromKnnMethodContext_luceneEngine() throws IOException {
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        KNNEngine.LUCENE.setInitialized(false);

        int efConstruction = 321;
        int m = 12;
        int dimension = 133;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2)
            .field(KNN_ENGINE, LUCENE_NAME)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
            .field(METHOD_PARAMETER_M, m)
            .endObject()
            .endObject()
            .endObject();
        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(indexName, settings)
        );
        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        builder.build(builderContext);

        assertEquals(METHOD_HNSW, builder.knnMethodContext.get().getMethodComponentContext().getName());
        assertEquals(
            efConstruction,
            builder.knnMethodContext.get().getMethodComponentContext().getParameters().get(METHOD_PARAMETER_EF_CONSTRUCTION)
        );
        assertTrue(KNNEngine.LUCENE.isInitialized());

        XContentBuilder xContentBuilderEmptyParams = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2)
            .field(KNN_ENGINE, LUCENE_NAME)
            .endObject()
            .endObject();
        KNNVectorFieldMapper.Builder builderEmptyParams = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilderEmptyParams),
            buildParserContext(indexName, settings)
        );

        assertEquals(METHOD_HNSW, builder.knnMethodContext.get().getMethodComponentContext().getName());
        assertTrue(builderEmptyParams.knnMethodContext.get().getMethodComponentContext().getParameters().isEmpty());

        XContentBuilder xContentBuilderUnsupportedParam = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2)
            .field(KNN_ENGINE, LUCENE_NAME)
            .startObject(PARAMETERS)
            .field("RANDOM_PARAM", 0)
            .endObject()
            .endObject()
            .endObject();

        expectThrows(
            ValidationException.class,
            () -> typeParser.parse(
                fieldName,
                xContentBuilderToMap(xContentBuilderUnsupportedParam),
                buildParserContext(indexName, settings)
            )
        );
    }

    @SneakyThrows
    public void testTypeParser_parse_compressionAndModeParameter() {
        String fieldName = "test-field-name-vec";
        String indexName = "test-index-name-vec";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        XContentBuilder xContentBuilder1 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION, 10)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x16.getName())
            .endObject();

        Mapper.Builder<?> builder = typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder1),
            buildParserContext(indexName, settings)
        );

        assertTrue(builder instanceof KNNVectorFieldMapper.Builder);
        assertEquals(Mode.ON_DISK.getName(), ((KNNVectorFieldMapper.Builder) builder).mode.get());
        assertEquals(CompressionLevel.x16.getName(), ((KNNVectorFieldMapper.Builder) builder).compressionLevel.get());

        XContentBuilder xContentBuilder2 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION, 10)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
            .field(MODE_PARAMETER, "invalid")
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x16.getName())
            .endObject();

        expectThrows(
            MapperParsingException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder2), buildParserContext(indexName, settings))
        );

        XContentBuilder xContentBuilder3 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION, 10)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
            .field(COMPRESSION_LEVEL_PARAMETER, "invalid")
            .endObject();

        expectThrows(
            MapperParsingException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder3), buildParserContext(indexName, settings))
        );

        XContentBuilder xContentBuilder4 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.DEFAULT.getValue())
            .field(MODE_PARAMETER, Mode.ON_DISK.getName())
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x16.getName())
            .endObject();

        expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder4), buildParserContext(indexName, settings))
        );
    }

    public void testTypeParser_parse_fromKnnMethodContext() throws IOException {
        // Check that knnMethodContext is set
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        int efConstruction = 321;
        int dimension = 133;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, DISK_ANN)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, efConstruction)
            .endObject()
            .endObject()
            .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(indexName, settings)
        );

        assertEquals(DISK_ANN, builder.knnMethodContext.get().getMethodComponentContext().getName());
        assertEquals(
            efConstruction,
            builder.knnMethodContext.get().getMethodComponentContext().getParameters().get(METHOD_PARAMETER_EF_CONSTRUCTION)
        );

        // Test invalid parameter
        XContentBuilder xContentBuilder2 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, DISK_ANN)
            .startObject(PARAMETERS)
            .field("invalid", "invalid")
            .endObject()
            .endObject()
            .endObject();

        expectThrows(
            ValidationException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder2), buildParserContext(indexName, settings))
        );

        // Test invalid method
        XContentBuilder xContentBuilder3 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, "invalid")
            .endObject()
            .endObject();

        expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder3), buildParserContext(indexName, settings))
        );

        // Test missing required parameter: dimension
        XContentBuilder xContentBuilder4 = XContentFactory.jsonBuilder().startObject().field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE).endObject();

        expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder4), buildParserContext(indexName, settings))
        );

        // Check that this fails if model id is also set
        XContentBuilder xContentBuilder5 = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .startObject(PARAMETERS)
            .field("invalid", "invalid")
            .endObject()
            .endObject()
            .endObject();

        expectThrows(
            IllegalArgumentException.class,
            () -> typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder5), buildParserContext(indexName, settings))
        );
    }

    public void testTypeParser_parse_fromLegacy() throws IOException {
        // Check that the particular values are set in builder
        String fieldName = "test-field-name";
        String indexName = "test-index-name";

        int m = 144;
        int efConstruction = 123;
        SpaceType spaceType = SpaceType.L2;
        Settings settings = Settings.builder().put(settings(CURRENT).build()).build();

        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();

        int dimension = 122;
        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .endObject();

        KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
            fieldName,
            xContentBuilderToMap(xContentBuilder),
            buildParserContext(indexName, settings)
        );

        assertNull(builder.knnMethodContext.get());
    }

    public IndexMetadata buildIndexMetaData(String indexName, Settings settings) {
        return IndexMetadata.builder(indexName)
            .settings(settings)
            .numberOfShards(1)
            .numberOfReplicas(0)
            .version(7)
            .mappingVersion(0)
            .settingsVersion(0)
            .aliasesVersion(0)
            .creationDate(0)
            .build();
    }

    public Mapper.TypeParser.ParserContext buildLegacyParserContext(String indexName, Settings settings, Version version) {
        return dobuildParserContext(indexName, settings, version);
    }

    public Mapper.TypeParser.ParserContext dobuildParserContext(String indexName, Settings settings, Version version) {
        IndexSettings indexSettings = new IndexSettings(
            buildIndexMetaData(indexName, settings),
            Settings.EMPTY,
            new IndexScopedSettings(Settings.EMPTY, new HashSet<>(IndexScopedSettings.BUILT_IN_INDEX_SETTINGS))
        );
        MapperService mapperService = mock(MapperService.class);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        // Setup blank
        return new Mapper.TypeParser.ParserContext(
            null,
            mapperService,
            type -> new KNNVectorFieldMapper.TypeParser(),
            version,
            null,
            null,
            null
        );
    }

    public Mapper.TypeParser.ParserContext buildParserContext(String indexName, Settings settings) {
        return dobuildParserContext(indexName, settings, CURRENT);
    }

    private static float[] createInitializedFloatArray(int dimension, float value) {
        float[] array = new float[dimension];
        Arrays.fill(array, value);
        return array;
    }

    private static byte[] createInitializedByteArray(int dimension, byte value) {
        byte[] array = new byte[dimension];
        Arrays.fill(array, value);
        return array;
    }

    public void testTypeParser_whenBinaryLuceneHNSW_thenValid() throws IOException {
        testTypeParserWithBinaryDataType(KNNEngine.LUCENE, SpaceType.HAMMING, METHOD_HNSW, 8, null);
    }

    public void testTypeParser_whenBinaryWithInvalidDimension_thenException() throws IOException {
        testTypeParserWithBinaryDataType(KNNEngine.LUCENE, SpaceType.HAMMING, METHOD_HNSW, 4, "should be multiply of 8");
    }

    public void testTypeParser_whenBinaryFaissHNSWWithInvalidSpaceType_thenException() throws IOException {
        for (SpaceType spaceType : SpaceType.values()) {
            if (SpaceType.UNDEFINED == spaceType || SpaceType.HAMMING == spaceType) {
                continue;
            }
            testTypeParserWithBinaryDataType(KNNEngine.LUCENE, spaceType, METHOD_HNSW, 8, "is not supported with");
        }
    }

    public void testTypeParser_whenBinaryLuceneHNSWWithSQ_thenException() throws IOException {
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();
        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, 8)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.BINARY.getValue())
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, KNNEngine.LUCENE.getName())
            .startObject(PARAMETERS)
            .startObject(METHOD_ENCODER_PARAMETER)
            .field(NAME, ENCODER_SQ)
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Exception ex = expectThrows(
            Exception.class,
            () -> typeParser.parse("test", xContentBuilderToMap(xContentBuilder), buildParserContext("test", settings))
        );
        assertTrue(ex.getMessage(), ex.getMessage().contains("parameter validation failed for MethodComponentContext parameter [encoder]"));
    }

    public void testBuilder_whenBinaryWithLegacyKNNDisabled_thenValid() {
        // Check legacy is picked up if model context and method context are not set
        KNNVectorFieldMapper.Builder builder = new KNNVectorFieldMapper.Builder("test-field-name-1", CURRENT, null, null);
        builder.vectorDataType.setValue(VectorDataType.BINARY);
        builder.dimension.setValue(8);

        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, false).build();

        builder.setOriginalParameters(new OriginalMappingParameters(builder));
        Mapper.BuilderContext builderContext = new Mapper.BuilderContext(settings, new ContentPath());
        KNNVectorFieldMapper knnVectorFieldMapper = builder.build(builderContext);
        assertTrue(knnVectorFieldMapper instanceof FlatVectorFieldMapper);
    }

    private void testTypeParserWithBinaryDataType(
        KNNEngine knnEngine,
        SpaceType spaceType,
        String method,
        int dimension,
        String expectedErrMsg
    ) throws IOException {
        // Check legacy is picked up if model context and method context are not set
        KNNVectorFieldMapper.TypeParser typeParser = new KNNVectorFieldMapper.TypeParser();
        String fieldName = "test-field-name-1";
        String indexName = "test-index";

        // Setup settings
        Settings settings = Settings.builder().put(settings(CURRENT).build()).put(KNN_INDEX, true).build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(TYPE_FIELD_NAME, KNN_VECTOR_TYPE)
            .field(DIMENSION_FIELD_NAME, dimension)
            .field(VECTOR_DATA_TYPE_FIELD, VectorDataType.BINARY.getValue())
            .startObject(KNN_METHOD)
            .field(NAME, method)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, knnEngine.getName())
            .endObject()
            .endObject();

        if (expectedErrMsg == null) {
            KNNVectorFieldMapper.Builder builder = (KNNVectorFieldMapper.Builder) typeParser.parse(
                fieldName,
                xContentBuilderToMap(xContentBuilder),
                buildParserContext(indexName, settings)
            );

            assertEquals(spaceType, builder.getOriginalParameters().getResolvedKnnMethodContext().getSpaceType());
        } else {
            Exception ex = expectThrows(Exception.class, () -> {
                typeParser.parse(fieldName, xContentBuilderToMap(xContentBuilder), buildParserContext(indexName, settings));
            });
            assertTrue(ex.getMessage(), ex.getMessage().contains(expectedErrMsg));
        }
    }
}

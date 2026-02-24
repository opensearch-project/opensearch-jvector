/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.common;

import org.opensearch.knn.index.VectorDataType;

public class KNNConstants {
    // shared across library constants
    public static final String DIMENSION = "dimension";
    public static final String KNN_ENGINE = "engine";
    public static final String KNN_METHOD = "method";
    public static final String NAME = "name";
    public static final String PARAMETERS = "parameters";
    public static final String METHOD_HNSW = "hnsw";
    public static final String DISK_ANN = "disk_ann";
    public static final String TYPE = "type";
    public static final String TYPE_NESTED = "nested";
    public static final String PATH = "path";
    public static final String QUERY = "query";
    public static final String KNN = "knn";
    public static final String VECTOR = "vector";
    public static final String K = "k";
    public static final String TYPE_KNN_VECTOR = "knn_vector";
    public static final String PROPERTIES = "properties";
    public static final String METHOD_PARAMETER = "method_parameters";
    public static final String METHOD_PARAMETER_EF_SEARCH = "ef_search";
    public static final String METHOD_PARAMETER_EF_CONSTRUCTION = "ef_construction";
    public static final String METHOD_PARAMETER_M = "m";
    public static final String METHOD_IVF = "ivf";
    public static final String METHOD_PARAMETER_NLIST = "nlist";
    public static final String METHOD_PARAMETER_SPACE_TYPE = "space_type"; // used for mapping parameter
    // used for defining toplevel parameter
    public static final String TOP_LEVEL_PARAMETER_SPACE_TYPE = METHOD_PARAMETER_SPACE_TYPE;
    public static final String COMPOUND_EXTENSION = "c";
    public static final String PLUGIN_NAME = "knn";
    public static final Integer BYTES_PER_KILOBYTES = 1024;

    public static final String MODEL_NODE_ASSIGNMENT = "training_node_assignment";
    public static final String MODEL_METHOD_COMPONENT_CONTEXT = "model_definition";

    public static final String QFRAMEWORK_CONFIG = "qframe_config";

    public static final String VECTOR_DATA_TYPE_FIELD = "data_type";
    public static final String EXPAND_NESTED = "expand_nested_docs";
    public static final String MODEL_VECTOR_DATA_TYPE_KEY = VECTOR_DATA_TYPE_FIELD;
    public static final VectorDataType DEFAULT_VECTOR_DATA_TYPE_FIELD = VectorDataType.FLOAT;
    public static final String MINIMAL_MODE_AND_COMPRESSION_FEATURE = "mode_and_compression_feature";
    public static final String TOP_LEVEL_SPACE_TYPE_FEATURE = "top_level_space_type_feature";

    public static final String RADIAL_SEARCH_KEY = "radial_search";
    public static final String MODEL_VERSION = "model_version";
    public static final String QUANTIZATION_STATE_FILE_SUFFIX = "osknnqstate";

    // MMR (Maximal Marginal Relevance) constants
    public static final String MMR = "mmr";
    public static final String DIVERSITY = "diversity";
    public static final String CANDIDATES = "candidates";
    public static final String VECTOR_FIELD_PATH = "vector_field_path";
    public static final String VECTOR_FIELD_DATA_TYPE = "vector_field_data_type";
    public static final String VECTOR_FIELD_SPACE_TYPE = "vector_field_space_type";
    public static final String MMR_RERANK_CONTEXT = "mmr.rerank_context";

    // Lucene specific constants
    public static final String LUCENE_NAME = "lucene";
    public static final String LUCENE_SQ_CONFIDENCE_INTERVAL = "confidence_interval";
    public static final int DYNAMIC_CONFIDENCE_INTERVAL = 0;
    public static final double MINIMUM_CONFIDENCE_INTERVAL = 0.9;
    public static final double MAXIMUM_CONFIDENCE_INTERVAL = 1.0;
    public static final String LUCENE_SQ_BITS = "bits";
    public static final int LUCENE_SQ_DEFAULT_BITS = 7;

    // nmslib specific constants
    public static final String SPACE_TYPE = "spaceType"; // used as field info key
    public static final String HNSW_ALGO_M = "M";
    public static final String HNSW_ALGO_EF_CONSTRUCTION = "efConstruction";

    // Faiss specific constants
    public static final String METHOD_ENCODER_PARAMETER = "encoder";
    public static final String METHOD_PARAMETER_NPROBES = "nprobes";
    public static final String ENCODER_FLAT = "flat";
    public static final String ENCODER_PARAMETER_PQ_M = "m";
    public static final String ENCODER_PARAMETER_PQ_CODE_SIZE = "code_size";
    public static final String ENCODER_SQ = "sq";

    /***** JVector specific constants *****/
    public static final String JVECTOR_NAME = "jvector";
    // Query related params
    public static final String METHOD_PARAMETER_OVERQUERY_FACTOR = "overquery_factor";
    public static final String METHOD_PARAMETER_THRESHOLD = "advanced.threshold";
    public static final String METHOD_PARAMETER_RERANK_FLOOR = "advanced.rerank_floor";
    public static final String METHOD_PARAMETER_USE_PRUNING = "advanced.use_pruning"; // TODO: wire this after jvector upgrade
    public static final int DEFAULT_OVER_QUERY_FACTOR = 5; // We will query 5x more than topKFor reranking
    public static final Double DEFAULT_QUERY_SIMILARITY_THRESHOLD = 0.0;
    public static final Double DEFAULT_QUERY_RERANK_FLOOR = 0.0;
    public static final Boolean DEFAULT_QUERY_USE_PRUNING = false; // TODO: wire this after jvector upgrade

    // Construction related params
    public static final String METHOD_PARAMETER_ALPHA = "advanced.alpha";
    public static final String METHOD_PARAMETER_NEIGHBOR_OVERFLOW = "advanced.neighbor_overflow";
    public static final String METHOD_PARAMETER_MIN_BATCH_SIZE_FOR_QUANTIZATION = "advanced.min_batch_size_for_quantization";
    public static final String METHOD_PARAMETER_HIERARCHY_ENABLED = "advanced.hierarchy_enabled";
    public static final String METHOD_PARAMETER_NUM_PQ_SUBSPACES = "advanced.num_pq_subspaces";
    public static final Double DEFAULT_ALPHA_VALUE = 1.2;
    public static final Double DEFAULT_NEIGHBOR_OVERFLOW_VALUE = 1.2;
    public static final int DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION = 1024; // above this batch size we will trigger quantization by
                                                                                // default
    public static final Boolean DEFAULT_HIERARCHY_ENABLED = false;

    // API Constants
    public static final String CLEAR_CACHE = "clear_cache";

    public static final Float DEFAULT_LUCENE_RADIAL_SEARCH_TRAVERSAL_SIMILARITY_RATIO = 0.95f;
    public static final String MIN_SCORE = "min_score";
    public static final String MAX_DISTANCE = "max_distance";

    public static final String MODE_PARAMETER = "mode";
    public static final String COMPRESSION_LEVEL_PARAMETER = "compression_level";

    public static final String DERIVED_VECTOR_FIELD_ATTRIBUTE_KEY = "knn-derived-source-enabled";
    public static final String DERIVED_VECTOR_FIELD_ATTRIBUTE_TRUE_VALUE = "true";
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Contains a map of counters to keep track of different values
 */
public enum KNNCounter {
    GRAPH_QUERY_ERRORS("graph_query_errors"),
    GRAPH_QUERY_REQUESTS("graph_query_requests"),
    GRAPH_INDEX_ERRORS("graph_index_errors"),
    GRAPH_INDEX_REQUESTS("graph_index_requests"),
    KNN_QUERY_REQUESTS("knn_query_requests"),
    SCRIPT_COMPILATIONS("script_compilations"),
    SCRIPT_COMPILATION_ERRORS("script_compilation_errors"),
    SCRIPT_QUERY_REQUESTS("script_query_requests"),
    SCRIPT_QUERY_ERRORS("script_query_errors"),
    TRAINING_REQUESTS("training_requests"),
    TRAINING_ERRORS("training_errors"),
    KNN_QUERY_WITH_FILTER_REQUESTS("knn_query_with_filter_requests"),
    MIN_SCORE_QUERY_REQUESTS("min_score_query_requests"),
    MIN_SCORE_QUERY_WITH_FILTER_REQUESTS("min_score_query_with_filter_requests"),
    MAX_DISTANCE_QUERY_REQUESTS("max_distance_query_requests"),
    MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS("max_distance_query_with_filter_requests"),
    KNN_QUERY_VISITED_NODES("knn_query_visited_nodes"),
    KNN_QUERY_RERANKED_COUNT("knn_query_reranked_count"),
    KNN_QUERY_EXPANDED_NODES("knn_query_expanded_nodes"),
    KNN_QUERY_EXPANDED_BASE_LAYER_NODES("knn_query_expanded_base_layer_nodes"),
    KNN_QUERY_GRAPH_SEARCH_TIME("knn_query_graph_search_time"), // The query time portion that is spent on scanning the vector graph
    KNN_QUANTIZATION_TRAINING_TIME("knn_quantization_training_time"), // The time in indexing/merges that is spent on training the
                                                                      // quantization parameters
    KNN_GRAPH_MERGE_TIME("knn_graph_merge_time"); // The time taken for jVector graph merges

    private final String name;
    private final AtomicLong count;

    /**
     * Constructor
     *
     * @param name name of the counter
     */
    KNNCounter(String name) {
        this.name = name;
        this.count = new AtomicLong(0);
    }

    /**
     * Get name of counter
     *
     * @return name
     */
    public String getName() {
        return name;
    }

    /**
     * Get the value of count
     *
     * @return count
     */
    public Long getCount() {
        return count.get();
    }

    /**
     * Increment the value of a counter
     */
    public void increment() {
        count.getAndIncrement();
    }

    /**
     * @param value counter value
     * Set the value of a counter
     */
    public void set(long value) {
        count.set(value);
    }

    /**
     * Adds the specified delta to the current value of the counter.
     *
     * @param delta the value to add to the counter
     */
    public void add(long delta) {
        count.addAndGet(delta);
    }
}

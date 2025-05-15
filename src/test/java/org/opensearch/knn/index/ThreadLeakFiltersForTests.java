/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.carrotsearch.randomizedtesting.ThreadFilter;

/**
 * Filter for ForkJoinPool worker threads used by JVector
 * This is used to ignore the ForkJoinPool worker threads used by JVector while still enabling thread leak detection
 */
public class ThreadLeakFiltersForTests implements ThreadFilter {
    @Override
    public boolean reject(Thread thread) {
        return thread.getName().startsWith("ForkJoinPool")
            && (thread.getThreadGroup().getName().contains("InternalKNNEngineTests")
                || thread.getThreadGroup().getName().startsWith("TGRP-KNNJVectorTests")
                || thread.getThreadGroup().getName().startsWith("TGRP-JVectorConcurrentQueryTests")
                || thread.getThreadGroup().getName().startsWith("TGRP-MemoryUsageAnalysisTests"));
    }
}

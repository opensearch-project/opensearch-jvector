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
        ThreadGroup threadGroup = thread.getThreadGroup();
        return thread.getName().startsWith("ForkJoinPool")
            && threadGroup != null
            && (threadGroup.getName().contains("InternalKNNEngineTests")
                || threadGroup.getName().startsWith("TGRP-KNNJVectorTests")
                || threadGroup.getName().startsWith("TGRP-JVectorConcurrentQueryTests")
                || threadGroup.getName().startsWith("TGRP-JVectorMergeWithDeletedDocsTests")
                || threadGroup.getName().startsWith("TGRP-MemoryUsageAnalysisTests"));
    }
}

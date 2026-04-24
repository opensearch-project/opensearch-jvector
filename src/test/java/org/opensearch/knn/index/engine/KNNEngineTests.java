/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.jvector.JVector;
import org.opensearch.knn.index.engine.lucene.Lucene;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.JVECTOR_NAME;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;

public class KNNEngineTests extends KNNTestCase {
    /**
     * Check that version from engine and library match
     */
    public void testDelegateLibraryFunctions() {
        assertEquals(JVector.INSTANCE.getVersion(), KNNEngine.JVECTOR.getVersion());
        assertEquals(Lucene.INSTANCE.getVersion(), KNNEngine.LUCENE.getVersion());
    }

    public void testGetDefaultEngine_thenReturnJVECTOR() {
        assertEquals(KNNEngine.JVECTOR, KNNEngine.DEFAULT);
    }

    /**
     * Test name getter
     */
    public void testGetName() {
        assertEquals(LUCENE_NAME, KNNEngine.LUCENE.getName());
        assertEquals(JVECTOR_NAME, KNNEngine.JVECTOR.getName());
    }

    /**
     * Test engine getter
     */
    public void testGetEngine() {
        assertEquals(KNNEngine.LUCENE, KNNEngine.getEngine(LUCENE_NAME));
        assertEquals(KNNEngine.JVECTOR, KNNEngine.getEngine(JVECTOR_NAME));
        expectThrows(IllegalArgumentException.class, () -> KNNEngine.getEngine("invalid"));
    }

    public void testGetEngineFromPath() {
        // In opensearch-jvector, getEngineNameFromPath always throws exception (line 76)
        String anyPath = "test.any";
        expectThrows(IllegalArgumentException.class, () -> KNNEngine.getEngineNameFromPath(anyPath));
    }

    public void testMmapFileExtensions() {
        final List<String> mmapExtensions = Arrays.stream(KNNEngine.values())
            .flatMap(engine -> engine.mmapFileExtensions().stream())
            .collect(Collectors.toList());
        assertNotNull(mmapExtensions);
        final List<String> expectedSettings = List.of("vex", "vec");
        assertTrue(expectedSettings.containsAll(mmapExtensions));
        assertTrue(mmapExtensions.containsAll(expectedSettings));
    }

    public void testGetEnginesThatCreateCustomSegmentFiles() {
        // opensearch-jvector returns empty set (line 85)
        assertTrue(KNNEngine.getEnginesThatCreateCustomSegmentFiles().isEmpty());
    }

    public void testGetEnginesThatSupportsFilters() {
        // Only LUCENE supports filters (line 32)
        assertEquals(1, KNNEngine.getEnginesThatSupportsFilters().size());
        assertTrue(KNNEngine.getEnginesThatSupportsFilters().contains(KNNEngine.LUCENE));
    }

    public void testGetMaxDimensionByEngine() {
        // Both engines support 16,000 dimensions (line 35)
        assertEquals(16_000, KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE));
        assertEquals(16_000, KNNEngine.getMaxDimensionByEngine(KNNEngine.JVECTOR));
    }
}

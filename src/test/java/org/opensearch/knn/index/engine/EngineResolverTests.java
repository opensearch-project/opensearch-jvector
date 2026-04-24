/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

public class EngineResolverTests extends KNNTestCase {

    private static final EngineResolver ENGINE_RESOLVER = EngineResolver.INSTANCE;

    public void testResolveEngine_whenEngineSpecifiedInMethod_thenThatEngine() {
        assertEquals(
            KNNEngine.LUCENE,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().build(),
                new KNNMethodContext(KNNEngine.LUCENE, SpaceType.DEFAULT, MethodComponentContext.EMPTY),
                false
            )
        );
    }

    public void testResolveEngine_whenRequiresTraining_thenJvector() {
        assertEquals(KNNEngine.JVECTOR, ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, true));
    }

    public void testResolveEngine_whenModeAndCompressionAreFalse_thenDefault() {
        assertEquals(KNNEngine.DEFAULT, ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, false));
        assertEquals(
            KNNEngine.DEFAULT,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().build(),
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY, false),
                false
            )
        );
    }

    public void testResolveEngine_whenModeSpecifiedAndCompressionIsNotSpecified_thenEngineBasedOnMode() {
        // When no mode or compression specified, returns DEFAULT
        assertEquals(KNNEngine.DEFAULT, ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().build(), null, false));

        // When mode is IN_MEMORY and compression not specified, returns LUCENE (line 52: mode != ON_DISK)
        assertEquals(
            KNNEngine.LUCENE,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).build(),
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY, false),
                false
            )
        );

        // When mode is ON_DISK and compression not specified, returns JVECTOR (line 52: mode == ON_DISK)
        assertEquals(
            KNNEngine.JVECTOR,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).build(),
                new KNNMethodContext(KNNEngine.DEFAULT, SpaceType.UNDEFINED, MethodComponentContext.EMPTY, false),
                false
            )
        );
    }

    public void testResolveEngine_whenCompressionIs1x_thenEngineBasedOnMode() {
        // When mode is ON_DISK with x1 compression, returns JVECTOR (line 52)
        assertEquals(
            KNNEngine.JVECTOR,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x1).build(),
                null,
                false
            )
        );

        // When mode is IN_MEMORY with x1 compression, returns LUCENE (line 52: mode != ON_DISK)
        assertEquals(
            KNNEngine.LUCENE,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).compressionLevel(CompressionLevel.x1).build(),
                null,
                false
            )
        );

        // When only x1 compression specified (no mode), defaults to LUCENE (line 52: mode != ON_DISK)
        assertEquals(
            KNNEngine.LUCENE,
            ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x1).build(), null, false)
        );
    }

    public void testResolveEngine_whenCompressionIs4x_thenEngineIsLucene() {
        assertEquals(
            KNNEngine.LUCENE,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x4).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.LUCENE,
            ENGINE_RESOLVER.resolveEngine(KNNMethodConfigContext.builder().compressionLevel(CompressionLevel.x4).build(), null, false)
        );
    }

    public void testResolveEngine_whenConfiguredForHighCompression_thenEngineIsJvector() {
        // For compressions other than x1 and x4, engine defaults to JVECTOR (line 60)

        // x2 compression
        assertEquals(
            KNNEngine.JVECTOR,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x2).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.JVECTOR,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).compressionLevel(CompressionLevel.x2).build(),
                null,
                false
            )
        );

        // x8 compression
        assertEquals(
            KNNEngine.JVECTOR,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x8).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.JVECTOR,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).compressionLevel(CompressionLevel.x8).build(),
                null,
                false
            )
        );

        // x16 compression
        assertEquals(
            KNNEngine.JVECTOR,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x16).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.JVECTOR,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).compressionLevel(CompressionLevel.x16).build(),
                null,
                false
            )
        );

        // x32 compression
        assertEquals(
            KNNEngine.JVECTOR,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.ON_DISK).compressionLevel(CompressionLevel.x32).build(),
                null,
                false
            )
        );
        assertEquals(
            KNNEngine.JVECTOR,
            ENGINE_RESOLVER.resolveEngine(
                KNNMethodConfigContext.builder().mode(Mode.IN_MEMORY).compressionLevel(CompressionLevel.x32).build(),
                null,
                false
            )
        );
    }
}

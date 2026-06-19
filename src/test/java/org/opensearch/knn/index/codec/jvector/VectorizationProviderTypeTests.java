/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.After;
import org.junit.Test;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeNoException;

public class VectorizationProviderTypeTests {

    private static final String NATIVE_VECTORIZATION_FEATURE_FLAG = "jvector.experimental.enable_native_vectorization";

    @After
    public void tearDown() {
        System.clearProperty(NATIVE_VECTORIZATION_FEATURE_FLAG);
    }

    @Test
    public void testGetByName_default_returnsCorrectProvider() {
        VectorizationProviderType provider = VectorizationProviderType.getByName("default");

        assertEquals(VectorizationProviderType.DEFAULT, provider);
    }

    @Test
    public void testGetByName_panama_returnsCorrectProvider() {
        VectorizationProviderType provider = VectorizationProviderType.getByName("panama");

        assertEquals(VectorizationProviderType.PANAMA, provider);
    }

    @Test
    public void testGetByName_native_returnsCorrectProvider() {
        VectorizationProviderType provider = VectorizationProviderType.getByName("native");

        assertEquals(VectorizationProviderType.NATIVE, provider);
    }

    @Test
    public void testGetByName_caseInsensitive_returnsCorrectProvider() {
        VectorizationProviderType provider = VectorizationProviderType.getByName("DEFAULT");

        assertEquals(VectorizationProviderType.DEFAULT, provider);
    }

    @Test
    public void testGetByName_unknownName_throwsIllegalArgumentException() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> VectorizationProviderType.getByName("unknown_provider")
        );

        assertTrue(exception.getMessage().contains("unknown_provider"));
    }

    @Test
    public void testGetByClassName_default_returnsCorrectProvider() {
        VectorizationProviderType provider = VectorizationProviderType.getByClassName(
            "io.github.jbellis.jvector.vector.DefaultVectorizationProvider"
        );

        assertEquals(VectorizationProviderType.DEFAULT, provider);
    }

    @Test
    public void testGetByClassName_panama_returnsCorrectProvider() {
        VectorizationProviderType provider = VectorizationProviderType.getByClassName(
            "io.github.jbellis.jvector.vector.PanamaVectorizationProvider"
        );

        assertEquals(VectorizationProviderType.PANAMA, provider);
    }

    @Test
    public void testGetByClassName_native_returnsCorrectProvider() {
        VectorizationProviderType provider = VectorizationProviderType.getByClassName(
            "io.github.jbellis.jvector.vector.NativeVectorizationProvider"
        );

        assertEquals(VectorizationProviderType.NATIVE, provider);
    }

    @Test
    public void testGetByClassName_unknownClassName_throwsIllegalArgumentException() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> VectorizationProviderType.getByClassName("com.example.UnknownProvider")
        );

        assertTrue(exception.getMessage().contains("com.example.UnknownProvider"));
    }

    @Test
    public void testGetVectorTypeSupport_default_returnsNonNullSupport() {
        VectorTypeSupport vectorTypeSupport = VectorizationProviderType.DEFAULT.getVectorTypeSupport();

        assertNotNull(vectorTypeSupport);
    }

    @Test
    public void testGetVectorTypeSupport_panama_returnsNonNullSupport() {
        VectorTypeSupport vectorTypeSupport = VectorizationProviderType.PANAMA.getVectorTypeSupport();

        assertNotNull(vectorTypeSupport);
    }

    @Test
    public void testGetVectorTypeSupport_default_isCached() {
        VectorTypeSupport first = VectorizationProviderType.DEFAULT.getVectorTypeSupport();
        VectorTypeSupport second = VectorizationProviderType.DEFAULT.getVectorTypeSupport();

        assertSame(first, second);
    }

    @Test
    public void testGetVectorTypeSupport_native_withoutFeatureFlag_throwsRuntimeException() {
        System.clearProperty(NATIVE_VECTORIZATION_FEATURE_FLAG);

        RuntimeException exception = assertThrows(RuntimeException.class, VectorizationProviderType.NATIVE::getVectorTypeSupport);

        assertTrue(exception.getMessage().contains(NATIVE_VECTORIZATION_FEATURE_FLAG));
    }

    @Test
    public void testGetVectorTypeSupport_native_withFeatureFlagEnabled_returnsNonNullSupport() {
        System.setProperty(NATIVE_VECTORIZATION_FEATURE_FLAG, "true");

        VectorTypeSupport vectorTypeSupport;
        try {
            vectorTypeSupport = VectorizationProviderType.NATIVE.getVectorTypeSupport();
        } catch (RuntimeException e) {
            // Native provider may be on the classpath but still unsupported on this machine
            // (e.g. missing AVX512), or unavailable for other environment-specific reasons.
            // Skip rather than fail in that case.
            assumeNoException("Native vectorization provider is not usable in this environment, skipping", e);
            return;
        }

        assertNotNull(vectorTypeSupport);
    }

    @Test
    public void testDefaultProvider_isNotNull() {
        assertNotNull(VectorizationProviderType.DEFAULT_PROVIDER);
    }

    @Test
    public void testGetName_returnsCorrectValues() {
        assertEquals("default", VectorizationProviderType.DEFAULT.getName());
        assertEquals("panama", VectorizationProviderType.PANAMA.getName());
        assertEquals("native", VectorizationProviderType.NATIVE.getName());
    }
}

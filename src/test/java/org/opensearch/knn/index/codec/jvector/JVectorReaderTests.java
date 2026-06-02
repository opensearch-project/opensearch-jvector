/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class JVectorReaderTests {

    private String originalProviderClassName;

    @Before
    public void setUp() {
        originalProviderClassName = JVectorReader.VectorizationProviderMapper.PROVIDER_CLASS_NAME;
    }

    @After
    public void tearDown() {
        JVectorReader.VectorizationProviderMapper.PROVIDER_CLASS_NAME = originalProviderClassName;
    }

    @Test
    public void testProviderToOrd_whenNativeProvider_returnsNativeOrdinal() {
        JVectorReader.VectorizationProviderMapper.PROVIDER_CLASS_NAME = "NativeVectorizationProvider";
        assertEquals(1, JVectorReader.VectorizationProviderMapper.providerToOrd());
    }

    @Test
    public void testProviderToOrd_whenNonNativeProvider_returnsNonNativeOrdinal() {
        JVectorReader.VectorizationProviderMapper.PROVIDER_CLASS_NAME = "DefaultVectorizationProvider";
        assertEquals(0, JVectorReader.VectorizationProviderMapper.providerToOrd());
    }

    @Test
    public void testOrdToProvider_whenNativeOrdinal_returnsNativeProvider() {
        JVectorReader.VectorizationProviderMapper.VectorizationProvider provider = JVectorReader.VectorizationProviderMapper.ordToProvider(
            1
        );

        assertEquals(JVectorReader.VectorizationProviderMapper.VectorizationProvider.NATIVE, provider);
    }

    @Test
    public void testOrdToProvider_whenNonNativeOrdinal_returnsNonNativeProvider() {
        JVectorReader.VectorizationProviderMapper.VectorizationProvider provider = JVectorReader.VectorizationProviderMapper.ordToProvider(
            0
        );

        assertEquals(JVectorReader.VectorizationProviderMapper.VectorizationProvider.NON_NATIVE, provider);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testOrdToProvider_whenNegativeOrdinal_throwsIllegalArgumentException() {
        JVectorReader.VectorizationProviderMapper.ordToProvider(-1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testOrdToProvider_whenOrdinalOutOfBounds_throwsIllegalArgumentException() {
        JVectorReader.VectorizationProviderMapper.ordToProvider(
            JVectorReader.VectorizationProviderMapper.JVECTOR_SUPPORTED_PROVIDERS.size()
        );
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

@Log4j2
public enum VectorizationProviderWrapper {
    // Default behavior for JVector library. Choose a provider based on jvector library lookup
    AUTO_DETECT(-1, "auto_detect", null, null),
    DEFAULT(0, "default", ByteOrder.BIG_ENDIAN, "io.github.jbellis.jvector.vector.DefaultVectorizationProvider"),
    PANAMA(1, "panama", ByteOrder.BIG_ENDIAN, "io.github.jbellis.jvector.vector.PanamaVectorizationProvider"),
    NATIVE(2, "native", ByteOrder.LITTLE_ENDIAN, "io.github.jbellis.jvector.vector.NativeVectorizationProvider");

    @Getter
    private final int id;
    @Getter
    private final String name;
    private final ByteOrder byteOrder;
    private final String className;
    private VectorTypeSupport vectorTypeSupport;

    private static final Map<String, VectorizationProviderWrapper> AVAILABLE_PROVIDERS_BY_NAME = new HashMap<>();
    private static final Map<String, VectorizationProviderWrapper> AVAILABLE_PROVIDERS_BY_CLASS = new HashMap<>();
    private static final Map<Integer, VectorizationProviderWrapper> AVAILABLE_PROVIDERS_BY_ID = new HashMap<>();
    private static final String NATIVE_VECTORIZATION_FEATURE_FLAG = "jvector.experimental.enable_native_vectorization";

    private static final VectorizationProviderWrapper RESOLVED_AUTO_DETECT;

    static {
        for (VectorizationProviderWrapper provider : values()) {
            AVAILABLE_PROVIDERS_BY_NAME.put(provider.name, provider);
            AVAILABLE_PROVIDERS_BY_ID.put(provider.id, provider);
            if (provider.className != null) {
                AVAILABLE_PROVIDERS_BY_CLASS.put(provider.className, provider);
            }
        }

        VectorizationProvider provider = VectorizationProvider.getInstance();
        VectorizationProviderWrapper resolved = getByClassName(provider.getClass().getName());
        if (resolved == null) {
            throw new IllegalStateException("Unsupported provider class: " + provider.getClass().getName());
        }
        RESOLVED_AUTO_DETECT = resolved;
    }

    VectorizationProviderWrapper(int id, String name, ByteOrder order, String className) {
        this.id = id;
        this.name = name;
        this.byteOrder = order;
        this.className = className;
    }

    public static VectorizationProviderWrapper getByName(String name) {
        return AVAILABLE_PROVIDERS_BY_NAME.get(name);
    }

    public static VectorizationProviderWrapper getByClassName(String className) {
        return AVAILABLE_PROVIDERS_BY_CLASS.get(className);
    }

    public static VectorizationProviderWrapper ordToProvider(int ord) {
        VectorizationProviderWrapper provider = AVAILABLE_PROVIDERS_BY_ID.get(ord);
        if (provider == null) {
            throw new IllegalArgumentException("Invalid id: " + ord);
        }
        return provider;
    }

    public VectorizationProviderWrapper resolve() {
        return this == AUTO_DETECT ? RESOLVED_AUTO_DETECT : this;
    }

    public synchronized VectorTypeSupport getVectorTypeSupport() {
        if (this == AUTO_DETECT) {
            throw new IllegalStateException("Call resolve() first - AUTO_DETECT has no vector type support");
        }

        if (vectorTypeSupport == null) {
            this.vectorTypeSupport = loadVectorTypeSupport();
        }

        return vectorTypeSupport;
    }

    public ByteOrder getByteOrder() {
        if (this == AUTO_DETECT) {
            throw new IllegalStateException("Call resolve() first - AUTO_DETECT has no byte order");
        }

        return byteOrder;
    }

    private VectorTypeSupport loadVectorTypeSupport() {
        if (this.equals(NATIVE) && !Boolean.getBoolean(NATIVE_VECTORIZATION_FEATURE_FLAG)) {
            throw new RuntimeException(
                String.format(
                    Locale.ROOT,
                    "Failed to load vector type support for %s. Property '%s' must be enabled",
                    className,
                    NATIVE_VECTORIZATION_FEATURE_FLAG
                )
            );
        }

        try {
            ClassLoader jvectorLoader = VectorizationProvider.class.getClassLoader();
            Class<?> clazz = Class.forName(className, true, jvectorLoader);
            Constructor<?> ctor = clazz.getConstructor();
            Object instance = ctor.newInstance();
            var provider = (VectorizationProvider) instance;
            return provider.getVectorTypeSupport();
        } catch (ClassNotFoundException | NoSuchMethodException | IllegalAccessException | InvocationTargetException
            | InstantiationException e) {
            throw new RuntimeException(String.format(Locale.ROOT, "Failed to load vector type support for %s", className), e);
        }
    }
}

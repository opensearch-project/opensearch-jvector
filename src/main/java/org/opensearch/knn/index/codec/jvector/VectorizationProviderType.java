/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import lombok.Getter;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public enum VectorizationProviderType {
    DEFAULT("default", ByteOrder.BIG_ENDIAN, "io.github.jbellis.jvector.vector.DefaultVectorizationProvider"),
    PANAMA("panama", ByteOrder.BIG_ENDIAN, "io.github.jbellis.jvector.vector.PanamaVectorizationProvider"),
    NATIVE("native", ByteOrder.LITTLE_ENDIAN, "io.github.jbellis.jvector.vector.NativeVectorizationProvider");

    @Getter
    private final String name;
    @Getter
    private final ByteOrder byteOrder;
    private final String className;
    private VectorTypeSupport vectorTypeSupport;

    private static final Map<String, VectorizationProviderType> AVAILABLE_PROVIDERS_BY_NAME = new HashMap<>();
    private static final Map<String, VectorizationProviderType> AVAILABLE_PROVIDERS_BY_CLASS = new HashMap<>();
    private static final String NATIVE_VECTORIZATION_FEATURE_FLAG = "jvector.experimental.enable_native_vectorization";

    /**
     * Resolved vectorization provider based on the lookup logic from datastax/jvector.
     * <a href="https://github.com/datastax/jvector/blob/a03dd608091a205fe9391c357414057115b1081f/jvector-base/src/main/java/io/github/jbellis/jvector/vector/VectorizationProvider.java#L79">VectorizationProvider</a>
     * */
    public static final VectorizationProviderType DEFAULT_PROVIDER;

    static {
        for (VectorizationProviderType provider : values()) {
            AVAILABLE_PROVIDERS_BY_NAME.put(provider.name, provider);
            if (provider.className != null) {
                AVAILABLE_PROVIDERS_BY_CLASS.put(provider.className, provider);
            }
        }

        io.github.jbellis.jvector.vector.VectorizationProvider provider = io.github.jbellis.jvector.vector.VectorizationProvider
            .getInstance();
        DEFAULT_PROVIDER = getByClassName(provider.getClass().getName());
    }

    VectorizationProviderType(String name, ByteOrder order, String className) {
        this.name = name;
        this.byteOrder = order;
        this.className = className;
    }

    public static VectorizationProviderType getByName(String name) {
        VectorizationProviderType provider = AVAILABLE_PROVIDERS_BY_NAME.get(name.toLowerCase(Locale.ROOT));
        if (provider != null) {
            return provider;
        }

        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT,
                "Unable to find provider with name: %s. Valid values are: %s",
                name,
                Arrays.toString(AVAILABLE_PROVIDERS_BY_NAME.keySet().toArray())
            )
        );
    }

    public static VectorizationProviderType getByClassName(String className) {
        VectorizationProviderType provider = AVAILABLE_PROVIDERS_BY_CLASS.get(className);
        if (provider != null) {
            return provider;
        }

        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT,
                "Unable to find provider with className: %s. Valid values are: %s",
                className,
                Arrays.toString(AVAILABLE_PROVIDERS_BY_CLASS.keySet().toArray())
            )
        );
    }

    public synchronized VectorTypeSupport getVectorTypeSupport() {
        if (vectorTypeSupport == null) {
            this.vectorTypeSupport = loadVectorTypeSupport();
        }

        return vectorTypeSupport;
    }

    private VectorTypeSupport loadVectorTypeSupport() {
        if (this == NATIVE && !Boolean.getBoolean(NATIVE_VECTORIZATION_FEATURE_FLAG)) {
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
            ClassLoader jvectorLoader = io.github.jbellis.jvector.vector.VectorizationProvider.class.getClassLoader();
            Class<?> clazz = Class.forName(className, true, jvectorLoader);
            Constructor<?> ctor = clazz.getConstructor();
            Object instance = ctor.newInstance();
            var provider = (io.github.jbellis.jvector.vector.VectorizationProvider) instance;
            return provider.getVectorTypeSupport();
        } catch (ClassNotFoundException | NoSuchMethodException | IllegalAccessException | InvocationTargetException
            | InstantiationException e) {
            throw new RuntimeException(String.format(Locale.ROOT, "Failed to load vector type support for %s", className), e);
        }
    }
}

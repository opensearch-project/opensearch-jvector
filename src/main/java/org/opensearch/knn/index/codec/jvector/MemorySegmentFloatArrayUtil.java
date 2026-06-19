/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;

/**
 * Converts a foreign-memory backed vector (java.lang.foreign.MemorySegment, Java 22+)
 * into a float[] using reflection and MethodHandles
 * <p>
 * Callers must check {@link #isSupported(Object)} before calling {@link #toFloatArray(Object, int)}.
 */
public final class MemorySegmentFloatArrayUtil {

    private static final Class<?> MEMORY_SEGMENT_CLASS;
    private static final MethodHandle TO_ARRAY_HANDLE;
    private static final Object JAVA_FLOAT_LAYOUT;

    static {
        Class<?> memorySegmentClass;
        MethodHandle toArrayHandle;
        Object javaFloatLayout;

        try {
            memorySegmentClass = Class.forName("java.lang.foreign.MemorySegment");
            Class<?> valueLayoutClass = Class.forName("java.lang.foreign.ValueLayout");
            Class<?> ofFloatClass = Class.forName("java.lang.foreign.ValueLayout$OfFloat");
            javaFloatLayout = valueLayoutClass.getField("JAVA_FLOAT").get(null);
            Method toArrayMethod = memorySegmentClass.getMethod("toArray", ofFloatClass);
            toArrayHandle = MethodHandles.lookup().unreflect(toArrayMethod);
        } catch (Throwable e) {
            memorySegmentClass = null;
            toArrayHandle = null;
            javaFloatLayout = null;
        }

        MEMORY_SEGMENT_CLASS = memorySegmentClass;
        TO_ARRAY_HANDLE = toArrayHandle;
        JAVA_FLOAT_LAYOUT = javaFloatLayout;
    }

    /**
     * Checks whether the MemorySegment#toArray API is available on this JVM
     * and whether backing is an instance of MemorySegment.
     *
     * @param backing the object backing the vector
     * @return true if toFloatArray can be safely called with this backing object
     */
    public static boolean isSupported(Object backing) {
        return MEMORY_SEGMENT_CLASS != null && MEMORY_SEGMENT_CLASS.isInstance(backing);
    }

    /**
     * Converts backing into a float[] via MemorySegment#toArray.
     * <p>
     * Callers must call {@link #isSupported(Object)} first; calling this method
     * with a backing object for which isSupported returned false is a programming error.
     *
     * @param backing the object backing the vector; must be a MemorySegment
     * @param length  the expected length of the resulting array (currently unused, kept for API clarity)
     * @return the converted float array
     * @throws RuntimeException if the underlying MethodHandle invocation fails
     */
    public static float[] toFloatArray(Object backing, int length) {
        try {
            return (float[]) TO_ARRAY_HANDLE.invoke(backing, JAVA_FLOAT_LAYOUT);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.io.IOException;

import static org.junit.Assert.assertArrayEquals;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class JVectorFloatVectorValuesTests {

    private static final int DIMENSION = 4;
    private static final int SIZE = 3;
    private static final float[] EXPECTED_FLOATS = { 1.0f, 2.0f, 3.0f, 4.0f };

    @Mock
    private OnDiskGraphIndex mockIndex;

    @Mock
    private OnDiskGraphIndex.View mockView;

    @Mock
    private GraphNodeIdToDocMap mockNodeMap;

    private JVectorFloatVectorValues vectorValues;

    @Before
    public void setUp() throws IOException {
        MockitoAnnotations.openMocks(this);

        when(mockIndex.getView()).thenReturn(mockView);
        when(mockView.dimension()).thenReturn(DIMENSION);
        when(mockView.size()).thenReturn(SIZE);
        when(mockView.liveNodes()).thenReturn(Bits.ALL);
        when(mockNodeMap.getMaxDoc()).thenReturn(SIZE);

        vectorValues = new JVectorFloatVectorValues(
            mockIndex,
            VectorSimilarityFunction.EUCLIDEAN,
            org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN,
            mockNodeMap,
            VectorizationProviderType.DEFAULT_PROVIDER.getVectorTypeSupport()
        );
    }

    @Test
    public void testVectorValue_afterFix_returnsCorrectFloatArray() throws IOException {
        VectorFloat<?> memorySegmentVector = createVectorFloatWithSegmentBacking(EXPECTED_FLOATS);
        doReturn(memorySegmentVector).when(mockView).getVector(0);

        float[] result = vectorValues.vectorValue(0);

        assertArrayEquals("vectorValue() should return correct float values", EXPECTED_FLOATS, result, 0.0f);
    }

    @Test
    public void testVectorValue_multipleOrdinals_returnsCorrectValues() throws IOException {
        float[] vector0 = { 1.0f, 0.0f, 0.0f, 0.0f };
        float[] vector1 = { 0.0f, 1.0f, 0.0f, 0.0f };
        float[] vector2 = { 0.0f, 0.0f, 1.0f, 0.0f };

        doReturn(createVectorFloatWithSegmentBacking(vector0)).when(mockView).getVector(0);
        doReturn(createVectorFloatWithSegmentBacking(vector1)).when(mockView).getVector(1);
        doReturn(createVectorFloatWithSegmentBacking(vector2)).when(mockView).getVector(2);

        assertArrayEquals(vector0, vectorValues.vectorValue(0), 0.0f);
        assertArrayEquals(vector1, vectorValues.vectorValue(1), 0.0f);
        assertArrayEquals(vector2, vectorValues.vectorValue(2), 0.0f);
    }

    @Test
    public void testVectorValue_zeroVector_returnsZeroArray() throws IOException {
        float[] zeros = new float[DIMENSION];
        doReturn(createVectorFloatWithSegmentBacking(zeros)).when(mockView).getVector(0);

        float[] result = vectorValues.vectorValue(0);

        assertArrayEquals(zeros, result, 0.0f);
    }

    @Test
    public void testVectorValue_withMockedVectorFloat_returnsCorrectValues() throws IOException {
        @SuppressWarnings("unchecked")
        VectorFloat<Object> mockedVector = mock(VectorFloat.class);
        when(mockedVector.length()).thenReturn(DIMENSION);
        for (int i = 0; i < DIMENSION; i++) {
            when(mockedVector.get(i)).thenReturn(EXPECTED_FLOATS[i]);
        }
        doReturn(mockedVector).when(mockView).getVector(0);

        float[] result = vectorValues.vectorValue(0);

        assertArrayEquals(EXPECTED_FLOATS, result, 0.0f);
    }

    @Test
    public void testVectorValue_withFloatArrayBacking_returnsCorrectValues() throws IOException {
        VectorFloat<?> arrayVector = createVectorFloatWithArrayBacking(EXPECTED_FLOATS);
        doReturn(arrayVector).when(mockView).getVector(0);

        float[] result = vectorValues.vectorValue(0);

        assertArrayEquals(EXPECTED_FLOATS, result, 0.0f);
    }

    @Test
    public void testVectorValue_withFloatArrayBacking_zeroVector_returnsZeroArray() throws IOException {
        float[] zeros = new float[DIMENSION];
        doReturn(createVectorFloatWithArrayBacking(zeros)).when(mockView).getVector(0);

        float[] result = vectorValues.vectorValue(0);

        assertArrayEquals(zeros, result, 0.0f);
    }

    @Test
    public void testVectorValue_withFloatArrayBacking_multipleOrdinals_returnsCorrectValues() throws IOException {
        float[] vector0 = { 1.0f, 0.0f, 0.0f, 0.0f };
        float[] vector1 = { 0.0f, 1.0f, 0.0f, 0.0f };
        float[] vector2 = { 0.0f, 0.0f, 1.0f, 0.0f };

        doReturn(createVectorFloatWithArrayBacking(vector0)).when(mockView).getVector(0);
        doReturn(createVectorFloatWithArrayBacking(vector1)).when(mockView).getVector(1);
        doReturn(createVectorFloatWithArrayBacking(vector2)).when(mockView).getVector(2);

        assertArrayEquals(vector0, vectorValues.vectorValue(0), 0.0f);
        assertArrayEquals(vector1, vectorValues.vectorValue(1), 0.0f);
        assertArrayEquals(vector2, vectorValues.vectorValue(2), 0.0f);
    }

    @SuppressWarnings("unchecked")
    private VectorFloat<?> createVectorFloatWithArrayBacking(float[] data) {
        VectorFloat<float[]> vector = mock(VectorFloat.class);
        when(vector.get()).thenReturn(data);
        when(vector.length()).thenReturn(data.length);
        for (int i = 0; i < data.length; i++) {
            when(vector.get(i)).thenReturn(data[i]);
        }
        return vector;
    }

    @SuppressWarnings("unchecked")
    private VectorFloat<?> createVectorFloatWithSegmentBacking(float[] data) {
        VectorFloat<Object> vector = mock(VectorFloat.class);
        when(vector.get()).thenReturn(new Object());
        when(vector.length()).thenReturn(data.length);
        for (int i = 0; i < data.length; i++) {
            when(vector.get(i)).thenReturn(data[i]);
        }
        return vector;
    }
}

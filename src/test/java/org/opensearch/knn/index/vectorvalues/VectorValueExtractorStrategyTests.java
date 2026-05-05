/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.SneakyThrows;
import org.apache.lucene.index.DocsWithFieldSet;
import org.junit.Assert;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;

import java.util.List;
import java.util.Map;

/**
 * To avoid unit test duplication, tests for exception is added here. For non exception cases tests are present in
 * {@link KNNVectorValuesTests}
 */
public class VectorValueExtractorStrategyTests extends KNNTestCase {
    final private List<float[]> floatArrayList = List.of(new float[] {1.3f, 2.2f, 3.2f}, new float[] {1.4f, 1.1f, 2.7f});
    final private List<byte[]> byteArrayList = List.of(new byte[] {1, 2, 3}, new byte[] {2, 3, 4});
    final List<byte[]> binaryArrayList = List.of(new byte[] {3, 2, 3}, new byte[] {1, 3, 2});

    @SneakyThrows
    public void testExtractWithDISI_whenInvalidIterator_thenException() {
        final VectorValueExtractorStrategy disiStrategy = new VectorValueExtractorStrategy.DISIVectorExtractor();
        final KNNVectorValuesIterator vectorValuesIterator = Mockito.mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        Mockito.when(vectorValuesIterator.getDocIdSetIterator()).thenReturn(new TestVectorValues.NotBinaryDocValues());
        Assert.assertThrows(IllegalArgumentException.class, () -> disiStrategy.extract(VectorDataType.FLOAT, vectorValuesIterator));
        Assert.assertThrows(IllegalArgumentException.class, () -> disiStrategy.extract(VectorDataType.BINARY, vectorValuesIterator));
        Assert.assertThrows(IllegalArgumentException.class, () -> disiStrategy.extract(VectorDataType.BYTE, vectorValuesIterator));
    }

    @SneakyThrows
    public void testExtractWithDISI_whenInvalidVectorDataType_thenException() {
        final TestVectorValues.PredefinedFloatVectorBinaryDocValues docValues = new TestVectorValues.PredefinedFloatVectorBinaryDocValues(floatArrayList);
        docValues.nextDoc();
        final VectorValueExtractorStrategy disiStrategy = new VectorValueExtractorStrategy.DISIVectorExtractor();
        final KNNVectorValuesIterator vectorValuesIterator = Mockito.mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        Mockito.when(vectorValuesIterator.getDocIdSetIterator()).thenReturn(docValues);
        Assert.assertThrows(IllegalArgumentException.class, () -> disiStrategy.extract(null, vectorValuesIterator));
    }

    @SneakyThrows
    public void testExtractWithDISI_extractFloatDocValues() {
        TestVectorValues.PredefinedFloatVectorBinaryDocValues vectorValues = new TestVectorValues.PredefinedFloatVectorBinaryDocValues(floatArrayList);
        KNNVectorValues<float[]> knnVectorValues = TestVectorValues.createKNNFloatVectorValues(vectorValues);
        new TestVectorValueExtractor<float[]>().testVectorValueExtractorDISIStrategy(VectorDataType.FLOAT, floatArrayList, knnVectorValues);
    }

    @SneakyThrows
    public void testExtractWithDISI_extractByteDocValues() {
        TestVectorValues.PredefinedByteVectorBinaryDocValues vectorValues = new TestVectorValues.PredefinedByteVectorBinaryDocValues(byteArrayList);
        KNNVectorValues<byte[]> knnVectorValues = TestVectorValues.createKNNBinaryVectorValues(vectorValues);
        new TestVectorValueExtractor<byte[]>().testVectorValueExtractorDISIStrategy(VectorDataType.BYTE, byteArrayList, knnVectorValues);
    }

    @SneakyThrows
    public void testExtractWithDISI_extractBinaryDocValues() {
        TestVectorValues.PredefinedByteVectorBinaryDocValues vectorValues = new TestVectorValues.PredefinedByteVectorBinaryDocValues(binaryArrayList);
        KNNVectorValues<byte[]> knnVectorValues = TestVectorValues.createKNNBinaryVectorValues(vectorValues);
        new TestVectorValueExtractor<byte[]>().testVectorValueExtractorDISIStrategy(VectorDataType.BINARY, binaryArrayList, knnVectorValues);
    }

    @SneakyThrows
    public void testExtractWithDISI_docIndexIterator_whenInvalidVectorDataType_thenException() {
        KNNVectorValues<float[]> vectorValues = TestVectorValues.createKNNFloatVectorValues(floatArrayList);
        final VectorValueExtractorStrategy disiStrategy = new VectorValueExtractorStrategy.DISIVectorExtractor();
        vectorValues.vectorValuesIterator.nextDoc();
        Assert.assertThrows(IllegalArgumentException.class, () -> disiStrategy.extract(null, vectorValues.vectorValuesIterator));
    }

    @SneakyThrows
    public void testExtractWithDISI_docIndexIterator_whenNoVectorOrDeletedDoc_thenReturnNull() {
        final VectorValueExtractorStrategy disiStrategy = new VectorValueExtractorStrategy.DISIVectorExtractor();
        final KNNVectorValuesIterator vectorValuesIterator = Mockito.mock(KNNVectorValuesIterator.DocIdsIteratorValues.class);
        Mockito.when(vectorValuesIterator.getDocIdSetIterator()).thenReturn( new TestVectorValues.NotExistingDocIndexIterator());
        byte[] result = disiStrategy.extract(VectorDataType.BYTE, vectorValuesIterator);
        Assert.assertNull(result);
    }

    @SneakyThrows
    public void testExtractWithDISI_docIndexIterator_extractFloatDocValues() {
        KNNVectorValues<float[]> vectorValues = TestVectorValues.createKNNFloatVectorValues(floatArrayList);
        new TestVectorValueExtractor<float[]>().testVectorValueExtractorDISIStrategy(VectorDataType.FLOAT, floatArrayList, vectorValues);
    }

    @SneakyThrows
    public void testExtractWithDISI_docIndexIterator_extractByteDocValues() {
        KNNVectorValues<byte[]> vectorValues = TestVectorValues.createKNNBinaryVectorValues(byteArrayList);
        new TestVectorValueExtractor<byte[]>().testVectorValueExtractorDISIStrategy(VectorDataType.BYTE, byteArrayList, vectorValues);
    }

    @SneakyThrows
    public void testExtractWithDISI_docIndexIterator_extractBinaryDocValues() {
        KNNVectorValues<byte[]> vectorValues = TestVectorValues.createKNNBinaryVectorValues(binaryArrayList);
        new TestVectorValueExtractor<byte[]>().testVectorValueExtractorDISIStrategy(VectorDataType.BINARY, binaryArrayList, vectorValues);
    }

    @SneakyThrows
    public void testExtractWithFieldWriter_docIndexIterator_extractFloatDocValues() {
        new TestVectorValueExtractor<float[]>().testVectorValueExtractorFieldWriterStrategy(VectorDataType.FLOAT, floatArrayList);
    }

    @SneakyThrows
    public void testExtractWithFieldWriter_docIndexIterator_extractByteDocValues() {
        new TestVectorValueExtractor<byte[]>().testVectorValueExtractorFieldWriterStrategy(VectorDataType.BYTE, byteArrayList);
    }

    @SneakyThrows
    public void testExtractWithFieldWriter_docIndexIterator_extractBinaryDocValues() {
        new TestVectorValueExtractor<byte[]>().testVectorValueExtractorFieldWriterStrategy(VectorDataType.BINARY, binaryArrayList);
    }

    private static class TestVectorValueExtractor<T> {
        @SneakyThrows
        private void testVectorValueExtractorDISIStrategy(VectorDataType vectorDataType, List<T> vectorValuesSourceList, KNNVectorValues<T> vectorValues) {
            final VectorValueExtractorStrategy disiStrategy = new VectorValueExtractorStrategy.DISIVectorExtractor();
            extractAndVerify(vectorDataType, vectorValuesSourceList, vectorValues, disiStrategy);
        }

        @SneakyThrows
        private void testVectorValueExtractorFieldWriterStrategy(VectorDataType vectorDataType, List<T> vectorValuesSourceList) {
            final VectorValueExtractorStrategy fieldWriterStrategy = new VectorValueExtractorStrategy.FieldWriterIteratorVectorExtractor();
            final DocsWithFieldSet docsWithFieldSet = TestVectorValues.getDocIdSetIterator(vectorValuesSourceList.size());
            final Map<Integer, T> vectorsMap = Map.of(0, vectorValuesSourceList.get(0), 1, vectorValuesSourceList.get(1));
            final KNNVectorValues<T> knnVectorValuesForFieldWriter = KNNVectorValuesFactory.getVectorValues(
                    vectorDataType,
                    docsWithFieldSet,
                    vectorsMap
            );
            extractAndVerify(vectorDataType, vectorValuesSourceList, knnVectorValuesForFieldWriter, fieldWriterStrategy);
        }

        @SneakyThrows
        private void extractAndVerify(VectorDataType vectorDataType, List<T> vectorValuesSourceList, KNNVectorValues<T> vectorValues, VectorValueExtractorStrategy extractorStrategy) {
            vectorValues.vectorValuesIterator.nextDoc();
            T result = extractorStrategy.extract(vectorDataType, vectorValues.vectorValuesIterator);
            T expected = vectorValuesSourceList.getFirst();
            Assert.assertArrayEquals(new Object[] { expected }, new Object[] { result });

            // Returns same vector if .nextDoc() is not triggered
            result = extractorStrategy.extract(vectorDataType, vectorValues.vectorValuesIterator);
            Assert.assertArrayEquals(new Object[] { expected }, new Object[] { result });

            vectorValues.vectorValuesIterator.nextDoc();
            expected = vectorValuesSourceList.get(1);
            result = extractorStrategy.extract(vectorDataType, vectorValues.vectorValuesIterator);
            Assert.assertArrayEquals(new Object[] { expected }, new Object[] { result });
        }
    }
}

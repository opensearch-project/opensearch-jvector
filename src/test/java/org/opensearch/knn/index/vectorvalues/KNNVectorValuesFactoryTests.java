/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.*;
import org.junit.Assert;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

public class KNNVectorValuesFactoryTests extends KNNTestCase {
    private static final int COUNT = 10;
    private static final int DIMENSION = 10;

    public void testGetVectorValuesFromDISI_whenValidInput_thenSuccess() {
        final BinaryDocValues binaryDocValues = new TestVectorValues.RandomVectorBinaryDocValues(COUNT, DIMENSION);
        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.FLOAT, binaryDocValues);
        Assert.assertNotNull(floatVectorValues);

        final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.BYTE, binaryDocValues);
        Assert.assertNotNull(byteVectorValues);

        final KNNVectorValues<byte[]> binaryVectorValues = KNNVectorValuesFactory.getVectorValues(VectorDataType.BINARY, binaryDocValues);
        Assert.assertNotNull(binaryVectorValues);
    }

    public void testGetVectorValuesUsingDocWithFieldSet_whenValidInput_thenSuccess() {
        final DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        docsWithFieldSet.add(0);
        docsWithFieldSet.add(1);
        final Map<Integer, float[]> floatVectorMap = Map.of(0, new float[] { 1, 2 }, 1, new float[] { 2, 3 });
        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            docsWithFieldSet,
            floatVectorMap
        );
        Assert.assertNotNull(floatVectorValues);

        final Map<Integer, byte[]> byteVectorMap = Map.of(0, new byte[] { 4, 5 }, 1, new byte[] { 6, 7 });

        final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.BYTE,
            docsWithFieldSet,
            byteVectorMap
        );
        Assert.assertNotNull(byteVectorValues);

        final KNNVectorValues<byte[]> binaryVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.BINARY,
            docsWithFieldSet,
            byteVectorMap
        );
        Assert.assertNotNull(binaryVectorValues);
    }

    public void testGetVectorValuesUsingDocValuesProducer_whenInvalidVectorEncoding_thenException() {
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final KnnVectorsReader reader = Mockito.mock(KnnVectorsReader.class);
        final TestVectorValues.ConstantVectorDocValuesProducer docValuesProducer = new TestVectorValues.ConstantVectorDocValuesProducer(
            3,
            2,
            3.2f
        );
        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(true);
        Mockito.when(fieldInfo.getName()).thenReturn("test_field");
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(null);

        Assert.assertThrows(
            IllegalArgumentException.class,
            () -> KNNVectorValuesFactory.getDerivedVectorValues(fieldInfo, docValuesProducer, reader)
        );
    }

    @SneakyThrows
    public void testGetVectorValuesUsingDocValuesProducer_whenValidInput_thenSuccess() {
        final List<float[]> floatArrayList = List.of(new float[] { 1.3f, 2.2f, 3.2f });
        final List<byte[]> byteArrayList = List.of(new byte[] { 1, 2, 3 });
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final KnnVectorsReader reader = Mockito.mock(KnnVectorsReader.class);
        final DocValuesProducer docValuesProducer = Mockito.mock(DocValuesProducer.class);
        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(true);
        Mockito.when(fieldInfo.getName()).thenReturn("test_field");

        // Checking for byte vectors
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);
        Mockito.when(reader.getByteVectorValues("test_field")).thenReturn(new TestVectorValues.PreDefinedByteVectorValues(byteArrayList));
        final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getDerivedVectorValues(
            fieldInfo,
            docValuesProducer,
            reader
        );
        byteVectorValues.nextDoc();
        Assert.assertArrayEquals(byteArrayList.getFirst(), byteVectorValues.getVector());

        // Checking for float vectors
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);
        Mockito.when(reader.getFloatVectorValues("test_field"))
            .thenReturn(new TestVectorValues.PreDefinedFloatVectorValues(floatArrayList));
        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getDerivedVectorValues(
            fieldInfo,
            docValuesProducer,
            reader
        );
        floatVectorValues.nextDoc();
        Assert.assertArrayEquals(floatArrayList.getFirst(), floatVectorValues.getVector(), 0.0f);

        // Checking for binary vectors with docValuesProducer
        // Note
        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(false);
        Mockito.when(reader.getByteVectorValues("test_field")).thenReturn(new TestVectorValues.PreDefinedByteVectorValues(byteArrayList));
        Mockito.when(docValuesProducer.getBinary(fieldInfo)).thenReturn(new TestVectorValues.ConstantVectorBinaryDocValues(1, 1, 2));
        final KNNVectorValues<float[]> binaryVectorValues = KNNVectorValuesFactory.getDerivedVectorValues(
            fieldInfo,
            docValuesProducer,
            reader
        );
        byteVectorValues.nextDoc();
        Assert.assertArrayEquals(new float[] { 2 }, binaryVectorValues.getVector(), 0);
    }

    public void testGetVectorValuesUsingDocValuesProducer_whenInvalidInput_thenException() {
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(false);
        Assert.assertThrows(IllegalArgumentException.class, () -> KNNVectorValuesFactory.getDerivedVectorValues(fieldInfo, null, null));

        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(true);
        Assert.assertThrows(IllegalArgumentException.class, () -> KNNVectorValuesFactory.getDerivedVectorValues(fieldInfo, null, null));
    }

    @SneakyThrows
    public void testGetVectorValuesFromFieldInfo_whenVectorDimIsNotZero_thenSuccess() {
        final List<byte[]> byteArrayList = List.of(new byte[] { 1, 2, 3 });
        final List<float[]> floatArrayList = List.of(new float[] { 1.3f, 2.2f, 3.2f });
        final List<byte[]> binaryArrayList = List.of(new byte[] { 3, 2, 3 });
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final SegmentReader reader = Mockito.mock(SegmentReader.class);
        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(true);
        Mockito.when(fieldInfo.getName()).thenReturn("test_field");

        // Checking for ByteVectorValues
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BYTE.getValue());
        Mockito.when(reader.getByteVectorValues("test_field")).thenReturn(new TestVectorValues.PreDefinedByteVectorValues(byteArrayList));
        final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        byteVectorValues.nextDoc();
        Assert.assertArrayEquals(byteArrayList.getFirst(), byteVectorValues.getVector());
        Assert.assertNotNull(byteVectorValues);

        // Checking for FloatVectorValues
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.FLOAT.getValue());
        Mockito.when(reader.getFloatVectorValues("test_field"))
            .thenReturn(new TestVectorValues.PreDefinedFloatVectorValues(floatArrayList));
        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        floatVectorValues.nextDoc();
        Assert.assertArrayEquals(floatArrayList.getFirst(), floatVectorValues.getVector(), 0.0f);
        Assert.assertNotNull(floatVectorValues);

        // Checking for BinaryVectorValues
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BINARY.getValue());
        Mockito.when(reader.getByteVectorValues("test_field"))
            .thenReturn(new TestVectorValues.PreDefinedBinaryVectorValues(binaryArrayList));
        final KNNVectorValues<byte[]> binaryVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        binaryVectorValues.nextDoc();
        Assert.assertArrayEquals(binaryArrayList.getFirst(), binaryVectorValues.getVector());
        Assert.assertNotNull(binaryVectorValues);
    }

    @SneakyThrows
    public void testGetVectorValuesFromFieldInfo_whenVectorEncodingInvalid_thenException() {
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final SegmentReader reader = Mockito.mock(SegmentReader.class);
        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(true);
        Mockito.when(fieldInfo.getName()).thenReturn("test_field");

        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(null);
        Assert.assertThrows(IllegalArgumentException.class, () -> KNNVectorValuesFactory.getVectorValues(fieldInfo, reader));
    }

    @SneakyThrows
    public void testGetVectorValuesFromFieldInfo_whenVectorDimIsZero_thenSuccess() {
        final List<byte[]> byteArrayList = List.of(new byte[] { 1, 2, 3 });
        final List<float[]> floatArrayList = List.of(new float[] { 1.3f, 2.2f, 3.2f });
        final List<byte[]> binaryArrayList = List.of(new byte[] { 3, 2, 3 });
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final SegmentReader reader = Mockito.mock(SegmentReader.class);
        Mockito.when(fieldInfo.hasVectorValues()).thenReturn(false);
        Mockito.when(fieldInfo.getName()).thenReturn("test_field");

        // Checking for ByteVectorValues
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BYTE.getValue());
        Mockito.when(reader.getBinaryDocValues("test_field"))
            .thenReturn(new TestVectorValues.PredefinedByteVectorBinaryDocValues(byteArrayList));

        final KNNVectorValues<byte[]> byteVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        byteVectorValues.nextDoc();
        Assert.assertArrayEquals(byteArrayList.getFirst(), byteVectorValues.getVector());
        Assert.assertNotNull(byteVectorValues);

        // Checking for Floats with BinaryDocValues
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.FLOAT.getValue());
        Mockito.when(reader.getBinaryDocValues("test_field"))
            .thenReturn(new TestVectorValues.PredefinedFloatVectorBinaryDocValues(floatArrayList));

        final KNNVectorValues<float[]> floatVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        floatVectorValues.nextDoc();
        Assert.assertArrayEquals(floatArrayList.getFirst(), floatVectorValues.getVector(), 0.0f);
        Assert.assertNotNull(floatVectorValues);

        // Checking for BinaryVectorValues
        Mockito.when(fieldInfo.getAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD)).thenReturn(VectorDataType.BINARY.getValue());
        Mockito.when(reader.getBinaryDocValues("test_field"))
            .thenReturn(new TestVectorValues.PredefinedByteVectorBinaryDocValues(binaryArrayList));

        final KNNVectorValues<byte[]> binaryVectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, reader);
        binaryVectorValues.nextDoc();
        Assert.assertArrayEquals(binaryArrayList.getFirst(), binaryVectorValues.getVector());
        Assert.assertNotNull(binaryVectorValues);

        Mockito.verify(fieldInfo, Mockito.times(0)).getVectorEncoding();
    }

    @SneakyThrows
    public void testGetVectorValuesSupplierFromDISI_whenValidInput_thenSuccess() {
        final List<float[]> floatArray = List.of(new float[] { 1, 2 }, new float[] { 2, 3 });
        final TestVectorValues.PreDefinedFloatVectorValues preDefinedFloatVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            floatArray
        );

        Supplier<KNNVectorValues<?>> supplier = KNNVectorValuesFactory.getVectorValuesSupplier(
            VectorDataType.FLOAT,
            preDefinedFloatVectorValues
        );

        final KNNVectorValues<?> knnVectorValues = supplier.get();
        final KNNVectorValuesIterator iterator = knnVectorValues.vectorValuesIterator;
        iterator.nextDoc();
        final float[] actual = (float[]) knnVectorValues.getVector();

        Assert.assertArrayEquals(floatArray.getFirst(), actual, 0.0f);
    }

    @SneakyThrows
    public void testGetVectorValuesSupplierUsingDocWithFieldSet_whenValidInput_thenSuccess() {
        final Map<Integer, float[]> floatVectorMap = Map.of(0, new float[] { 1, 2 }, 1, new float[] { 2, 3 });
        final DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        docsWithFieldSet.add(0);
        docsWithFieldSet.add(1);

        Supplier<KNNVectorValues<?>> supplier = KNNVectorValuesFactory.getVectorValuesSupplier(
            VectorDataType.FLOAT,
            docsWithFieldSet,
            floatVectorMap
        );

        final KNNVectorValues<?> knnVectorValues = supplier.get();
        final KNNVectorValuesIterator iterator = knnVectorValues.vectorValuesIterator;
        iterator.nextDoc();
        final float[] actual = (float[]) knnVectorValues.getVector();

        Assert.assertArrayEquals(floatVectorMap.get(0), actual, 0.0f);
    }

    @SneakyThrows
    public void testGetKNNVectorValuesSupplierForMerge_whenFloatVectors_thenSuccess() {
        final List<float[]> floatArray = List.of(new float[] { 1.0f, 2.0f }, new float[] { 3.0f, 4.0f });
        final TestVectorValues.PreDefinedFloatVectorValues preDefinedFloatVectorValues = new TestVectorValues.PreDefinedFloatVectorValues(
            floatArray
        );

        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final MergeState mergeState = Mockito.mock(MergeState.class);

        Mockito.when(fieldInfo.getName()).thenReturn("test_field");
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);

        try (
            MockedStatic<KnnVectorsWriter.MergedVectorValues> mockedStatic = Mockito.mockStatic(KnnVectorsWriter.MergedVectorValues.class)
        ) {
            mockedStatic.when(() -> KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState))
                .thenReturn(preDefinedFloatVectorValues);

            Supplier<KNNVectorValues<?>> supplier = KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(
                VectorDataType.FLOAT,
                fieldInfo,
                mergeState
            );

            Assert.assertNotNull(supplier);
            final KNNVectorValues<?> knnVectorValues = supplier.get();
            Assert.assertNotNull(knnVectorValues);
            Assert.assertTrue(knnVectorValues instanceof KNNFloatVectorValues);
        }
    }

    @SneakyThrows
    public void testGetKNNVectorValuesSupplierForMerge_whenByteVectors_thenSuccess() {
        final List<byte[]> byteArray = List.of(new byte[] { 1, 2 }, new byte[] { 3, 4 });
        final TestVectorValues.PreDefinedByteVectorValues preDefinedByteVectorValues = new TestVectorValues.PreDefinedByteVectorValues(
            byteArray
        );

        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final MergeState mergeState = Mockito.mock(MergeState.class);

        Mockito.when(fieldInfo.getName()).thenReturn("test_field");
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.BYTE);

        try (
            MockedStatic<KnnVectorsWriter.MergedVectorValues> mockedStatic = Mockito.mockStatic(KnnVectorsWriter.MergedVectorValues.class)
        ) {
            mockedStatic.when(() -> KnnVectorsWriter.MergedVectorValues.mergeByteVectorValues(fieldInfo, mergeState))
                .thenReturn(preDefinedByteVectorValues);

            Supplier<KNNVectorValues<?>> supplier = KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(
                VectorDataType.BYTE,
                fieldInfo,
                mergeState
            );

            Assert.assertNotNull(supplier);
            final KNNVectorValues<?> knnVectorValues = supplier.get();
            Assert.assertNotNull(knnVectorValues);
            Assert.assertTrue(knnVectorValues instanceof KNNByteVectorValues);
        }
    }

    @SneakyThrows
    public void testGetKNNVectorValuesSupplierForMerge_whenIOException_thenWrappedInIllegalStateException() {
        final FieldInfo fieldInfo = Mockito.mock(FieldInfo.class);
        final MergeState mergeState = Mockito.mock(MergeState.class);

        Mockito.when(fieldInfo.getName()).thenReturn("test_field");
        Mockito.when(fieldInfo.getVectorEncoding()).thenReturn(VectorEncoding.FLOAT32);

        try (
            MockedStatic<KnnVectorsWriter.MergedVectorValues> mockedStatic = Mockito.mockStatic(KnnVectorsWriter.MergedVectorValues.class)
        ) {
            mockedStatic.when(() -> KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState))
                .thenThrow(new IOException("Test IO error"));

            Supplier<KNNVectorValues<?>> supplier = KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(
                VectorDataType.FLOAT,
                fieldInfo,
                mergeState
            );

            Assert.assertNotNull(supplier);
            IllegalStateException exception = Assert.assertThrows(IllegalStateException.class, supplier::get);
            Assert.assertTrue(exception.getMessage().contains("Unable to merge vectors for field"));
            Assert.assertTrue(exception.getCause() instanceof IOException);
        }
    }
}

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.sandbox.codecs.faiss;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.apache.lucene.index.VectorSimilarityFunction.COSINE;
import static org.apache.lucene.index.VectorSimilarityFunction.DOT_PRODUCT;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.sandbox.codecs.faiss.FaissNativeWrapper.Exception.handleException;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.ByteOrder;
import java.util.Map;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.hnsw.IntToIntFunction;

/**
 * A native implementation of {@link FaissLibrary} using {@link FaissNativeWrapper}.
 */
@SuppressWarnings("restricted")
final class FaissLibraryNativeImpl implements FaissLibrary {
  private static final Logger logger = LogManager.getLogger(FaissLibraryNativeImpl.class);

  private final FaissNativeWrapper wrapper;

  FaissLibraryNativeImpl() {
    this.wrapper = new FaissNativeWrapper();
  }

  private static MemorySegment getStub(
      Arena arena, MethodHandle target, FunctionDescriptor descriptor) {
    return Linker.nativeLinker().upcallStub(target, descriptor, arena);
  }

  private static final int BUFFER_SIZE = 256 * 1024 * 1024; // 256 MB

  @SuppressWarnings("unused")
  private static long writeBytes(
      IndexOutput output, MemorySegment inputPointer, long itemSize, long numItems)
      throws IOException {
    long size = itemSize * numItems;
    inputPointer = inputPointer.reinterpret(size);

    if (size <= BUFFER_SIZE) {
      output.writeBytes(inputPointer.toArray(JAVA_BYTE), (int) size);
    } else {
      byte[] bytes = new byte[BUFFER_SIZE];
      for (long offset = 0; offset < size; offset += BUFFER_SIZE) {
        int length = (int) Math.min(size - offset, BUFFER_SIZE);
        MemorySegment.copy(inputPointer, JAVA_BYTE, offset, bytes, 0, length);
        output.writeBytes(bytes, length);
      }
    }
    return numItems;
  }

  @SuppressWarnings("unused")
  private static long readBytes(
      IndexInput input, MemorySegment outputPointer, long itemSize, long numItems)
      throws IOException {
    long size = itemSize * numItems;
    outputPointer = outputPointer.reinterpret(size);

    if (size <= BUFFER_SIZE) {
      byte[] bytes = new byte[(int) size];
      input.readBytes(bytes, 0, bytes.length);
      MemorySegment.copy(bytes, 0, outputPointer, JAVA_BYTE, 0, bytes.length);
    } else {
      byte[] bytes = new byte[BUFFER_SIZE];
      for (long offset = 0; offset < size; offset += BUFFER_SIZE) {
        int length = (int) Math.min(size - offset, BUFFER_SIZE);
        input.readBytes(bytes, 0, length);
        MemorySegment.copy(bytes, 0, outputPointer, JAVA_BYTE, offset, length);
      }
    }
    return numItems;
  }

  private static final MethodHandle WRITE_BYTES_HANDLE;
  private static final MethodHandle READ_BYTES_HANDLE;

  static {
    try {
      MethodHandles.Lookup lookup = MethodHandles.lookup();

      WRITE_BYTES_HANDLE =
          lookup.findStatic(
              FaissLibraryNativeImpl.class,
              "writeBytes",
              MethodType.methodType(
                  long.class, IndexOutput.class, MemorySegment.class, long.class, long.class));

      READ_BYTES_HANDLE =
          lookup.findStatic(
              FaissLibraryNativeImpl.class,
              "readBytes",
              MethodType.methodType(
                  long.class, IndexInput.class, MemorySegment.class, long.class, long.class));

    } catch (NoSuchMethodException | IllegalAccessException e) {
      throw new LinkageError(
          "FaissLibraryNativeImpl reader / writer functions are missing or inaccessible", e);
    }
  }

  private static final Map<VectorSimilarityFunction, Integer> FUNCTION_TO_METRIC =
      Map.of(
          DOT_PRODUCT, 0,
          COSINE, 0,
          EUCLIDEAN, 1);

  private static int functionToMetric(VectorSimilarityFunction function) {
    Integer metric = FUNCTION_TO_METRIC.get(function);
    if (metric == null) {
      throw new UnsupportedOperationException("Similarity function not supported: " + function);
    }
    return metric;
  }

  @Override
  public FaissLibrary.Index createIndex(
      String description,
      String indexParams,
      VectorSimilarityFunction function,
      FloatVectorValues floatVectorValues,
      IntToIntFunction oldToNewDocId) {

    try (Arena temp = Arena.ofConfined()) {
      int size = floatVectorValues.size();
      int dimension = floatVectorValues.dimension();
      int metric = functionToMetric(function);

      MemorySegment pointer = temp.allocate(ADDRESS);
      handleException(
          wrapper.faiss_index_factory(pointer, dimension, temp.allocateFrom(description), metric));

      MemorySegment indexPointer = pointer.get(ADDRESS, 0);

      handleException(wrapper.faiss_ParameterSpace_new(pointer));
      MemorySegment parameterSpacePointer =
          pointer
              .get(ADDRESS, 0)
              .reinterpret(temp, wrapper::faiss_ParameterSpace_free);

      handleException(
          wrapper.faiss_ParameterSpace_set_index_parameters(
              parameterSpacePointer, indexPointer, temp.allocateFrom(indexParams)));

      MemorySegment docs = temp.allocate(JAVA_FLOAT, (long) size * dimension);
      long docsOffset = 0;
      long perDocByteSize = dimension * JAVA_FLOAT.byteSize();

      MemorySegment ids = temp.allocate(JAVA_LONG, size);
      int idsIndex = 0;

      KnnVectorValues.DocIndexIterator iterator = floatVectorValues.iterator();
      for (int i = iterator.nextDoc(); i != NO_MORE_DOCS; i = iterator.nextDoc()) {
        int id = oldToNewDocId.apply(i);
        ids.setAtIndex(JAVA_LONG, idsIndex, id);
        idsIndex++;

        float[] vector = floatVectorValues.vectorValue(iterator.index());
        MemorySegment.copy(vector, 0, docs, JAVA_FLOAT, docsOffset, vector.length);
        docsOffset += perDocByteSize;
      }

      if (function == COSINE) {
        wrapper.faiss_fvec_renorm_L2(dimension, size, docs);
      }

      int isTrained = wrapper.faiss_Index_is_trained(indexPointer);
      if (isTrained == 0) {
        handleException(wrapper.faiss_Index_train(indexPointer, size, docs));
      }

      handleException(wrapper.faiss_Index_add_with_ids(indexPointer, size, docs, ids));

      return new Index(indexPointer, function);

    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static final int FAISS_IO_FLAG_MMAP = 1;
  private static final int FAISS_IO_FLAG_READ_ONLY = 2;

  @Override
  public FaissLibrary.Index readIndex(IndexInput input, VectorSimilarityFunction function) {
    try (Arena temp = Arena.ofConfined()) {
      MethodHandle readerHandle = READ_BYTES_HANDLE.bindTo(input);
      MemorySegment readerStub =
          getStub(
              temp, readerHandle, FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_LONG));

      MemorySegment pointer = temp.allocate(ADDRESS);
      handleException(wrapper.faiss_CustomIOReader_new(pointer, readerStub));
      MemorySegment customIOReaderPointer =
          pointer
              .get(ADDRESS, 0)
              .reinterpret(temp, wrapper::faiss_CustomIOReader_free);

      handleException(
          wrapper.faiss_read_index_custom(
              customIOReaderPointer, FAISS_IO_FLAG_MMAP | FAISS_IO_FLAG_READ_ONLY, pointer));
      MemorySegment indexPointer = pointer.get(ADDRESS, 0);

      return new Index(indexPointer, function);
    }
  }

  private class Index implements FaissLibrary.Index {
    @FunctionalInterface
    private interface FloatToFloatFunction {
      float scale(float score);
    }

    private final Arena arena;
    private final MemorySegment indexPointer;
    private final FloatToFloatFunction scaler;
    private final VectorSimilarityFunction function;
    private final int dimension;
    private boolean closed;

    private Index(MemorySegment indexPointer, VectorSimilarityFunction function) {
      this.arena = Arena.ofShared();
      this.indexPointer =
          indexPointer
              .reinterpret(arena, wrapper::faiss_Index_free);
      this.function = function;
      this.dimension = wrapper.faiss_Index_d(indexPointer);

      this.scaler =
          switch (function) {
            case DOT_PRODUCT, COSINE ->
                distance -> Math.max((1 + distance) / 2, 0);

            case EUCLIDEAN ->
                distance -> 1 / (1 + distance);

            case MAXIMUM_INNER_PRODUCT ->
                throw new UnsupportedOperationException(
                    "Similarity function not supported: " + function);
          };

      this.closed = false;
    }

    @Override
    public void close() {
      if (closed == false) {
        arena.close();
        closed = true;
      }
    }

    @Override
    public void search(float[] query, KnnCollector knnCollector, AcceptDocs acceptDocs) {
      try (Arena temp = Arena.ofConfined()) {
        FixedBitSet fixedBitSet =
            switch (acceptDocs.bits()) {
              case null -> null;
              case FixedBitSet bitSet -> bitSet;
              case Bits bits -> FixedBitSet.copyOf(bits);
            };

        MemorySegment queries = temp.allocateFrom(JAVA_FLOAT, query);
        if (function == COSINE) {
          wrapper.faiss_fvec_renorm_L2(dimension, 1, queries);
        }

        int k = knnCollector.k();
        MemorySegment distancesPointer = temp.allocate(JAVA_FLOAT, k);
        MemorySegment idsPointer = temp.allocate(JAVA_LONG, k);

        MemorySegment localIndex = indexPointer.reinterpret(temp, null);
        if (fixedBitSet == null) {
          logger.info("[FAISS NATIVE ENGINE] Calling native faiss_Index_search via Panama FFM (k={})", k);
          handleException(
              wrapper.faiss_Index_search(localIndex, 1, queries, k, distancesPointer, idsPointer));
        } else {
          logger.info("[FAISS NATIVE ENGINE] Calling native faiss_Index_search_with_params (filtered) via Panama FFM (k={})", k);
          MemorySegment pointer = temp.allocate(ADDRESS);

          long[] bits = fixedBitSet.getBits();
          MemorySegment nativeBits =
              temp.allocateFrom(JAVA_LONG.withOrder(ByteOrder.LITTLE_ENDIAN), bits);

          handleException(
              wrapper.faiss_IDSelectorBitmap_new(pointer, fixedBitSet.length(), nativeBits));
          MemorySegment idSelectorBitmapPointer =
              pointer
                  .get(ADDRESS, 0)
                  .reinterpret(temp, wrapper::faiss_IDSelectorBitmap_free);

          handleException(wrapper.faiss_SearchParameters_new(pointer, idSelectorBitmapPointer));
          MemorySegment searchParametersPointer =
              pointer
                  .get(ADDRESS, 0)
                  .reinterpret(temp, wrapper::faiss_SearchParameters_free);

          handleException(
              wrapper.faiss_Index_search_with_params(
                  localIndex,
                  1,
                  queries,
                  k,
                  searchParametersPointer,
                  distancesPointer,
                  idsPointer));
        }

        int resultCount = 0;
        for (int i = 0; i < k; i++) {
          int id = (int) idsPointer.getAtIndex(JAVA_LONG, i);

          if (id == -1) {
            break;
          }

          resultCount++;
          float distance = distancesPointer.getAtIndex(JAVA_FLOAT, i);
          float score = scaler.scale(distance);
          logger.info("[FAISS NATIVE ENGINE] Native hit #{}: docId={}, rawDistance={}, score={}", resultCount, id, distance, score);
          knnCollector.collect(id, score);
        }
        logger.info("[FAISS NATIVE ENGINE] Search completed. Total matched docs: {}", resultCount);
        if (resultCount > 0) {
          knnCollector.incVisitedCount(resultCount);
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    @Override
    public void write(IndexOutput output) {
      try (Arena temp = Arena.ofConfined()) {
        MethodHandle writerHandle = WRITE_BYTES_HANDLE.bindTo(output);
        MemorySegment writerStub =
            getStub(
                temp,
                writerHandle,
                FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_LONG));

        MemorySegment pointer = temp.allocate(ADDRESS);
        handleException(wrapper.faiss_CustomIOWriter_new(pointer, writerStub));
        MemorySegment customIOWriterPointer =
            pointer
                .get(ADDRESS, 0)
                .reinterpret(temp, wrapper::faiss_CustomIOWriter_free);

        handleException(
            wrapper.faiss_write_index_custom(
                indexPointer, customIOWriterPointer, FAISS_IO_FLAG_MMAP | FAISS_IO_FLAG_READ_ONLY));
      }
    }
  }
}

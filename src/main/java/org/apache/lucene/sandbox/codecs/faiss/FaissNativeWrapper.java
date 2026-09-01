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
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.Locale;

/**
 * Utility class to wrap necessary functions of the native C API of Faiss
 * using Project Panama.
 */
@SuppressWarnings("restricted")
final class FaissNativeWrapper {
  static {
    System.loadLibrary(FaissLibrary.NAME);
  }

  private static MethodHandle getHandle(String functionName, FunctionDescriptor descriptor) {
    MemorySegment symbol = SymbolLookup.loaderLookup().find(functionName).orElseThrow(
        () -> new UnsatisfiedLinkError("Symbol not found: " + functionName)
    );
    return Linker.nativeLinker().downcallHandle(symbol, descriptor);
  }

  FaissNativeWrapper() {
    // Check Faiss version
    String expectedVersion = FaissLibrary.VERSION;
    String actualVersion = faiss_get_version().reinterpret(Long.MAX_VALUE).getString(0);

    if (expectedVersion.equals(actualVersion) == false) {
      throw new LinkageError(
          String.format(
              Locale.ROOT,
              "Expected Faiss library version %s, found %s",
              expectedVersion,
              actualVersion));
    }
  }

  private final MethodHandle faiss_get_version$MH =
      getHandle("faiss_get_version", FunctionDescriptor.of(ADDRESS));

  MemorySegment faiss_get_version() {
    try {
      return (MemorySegment) faiss_get_version$MH.invokeExact();
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_CustomIOReader_free$MH =
      getHandle("faiss_CustomIOReader_free", FunctionDescriptor.ofVoid(ADDRESS));

  void faiss_CustomIOReader_free(MemorySegment customIOReaderPointer) {
    try {
      faiss_CustomIOReader_free$MH.invokeExact(customIOReaderPointer);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_CustomIOReader_new$MH =
      getHandle("faiss_CustomIOReader_new", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

  int faiss_CustomIOReader_new(MemorySegment pointer, MemorySegment readerStub) {
    try {
      return (int) faiss_CustomIOReader_new$MH.invokeExact(pointer, readerStub);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_CustomIOWriter_free$MH =
      getHandle("faiss_CustomIOWriter_free", FunctionDescriptor.ofVoid(ADDRESS));

  void faiss_CustomIOWriter_free(MemorySegment customIOWriterPointer) {
    try {
      faiss_CustomIOWriter_free$MH.invokeExact(customIOWriterPointer);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_CustomIOWriter_new$MH =
      getHandle("faiss_CustomIOWriter_new", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

  int faiss_CustomIOWriter_new(MemorySegment pointer, MemorySegment writerStub) {
    try {
      return (int) faiss_CustomIOWriter_new$MH.invokeExact(pointer, writerStub);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_IDSelectorBitmap_free$MH =
      getHandle("faiss_IDSelectorBitmap_free", FunctionDescriptor.ofVoid(ADDRESS));

  void faiss_IDSelectorBitmap_free(MemorySegment idSelectorBitmapPointer) {
    try {
      faiss_IDSelectorBitmap_free$MH.invokeExact(idSelectorBitmapPointer);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_IDSelectorBitmap_new$MH =
      getHandle("faiss_IDSelectorBitmap_new", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));

  int faiss_IDSelectorBitmap_new(MemorySegment pointer, int n, MemorySegment bitmap) {
    try {
      return (int) faiss_IDSelectorBitmap_new$MH.invokeExact(pointer, n, bitmap);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_Index_add_with_ids$MH =
      getHandle("faiss_Index_add_with_ids", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));

  int faiss_Index_add_with_ids(
      MemorySegment indexPointer, int n, MemorySegment docs, MemorySegment ids) {
    try {
      return (int) faiss_Index_add_with_ids$MH.invokeExact(indexPointer, n, docs, ids);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_Index_d$MH =
      getHandle("faiss_Index_d", FunctionDescriptor.of(JAVA_INT, ADDRESS));

  int faiss_Index_d(MemorySegment indexPointer) {
    try {
      return (int) faiss_Index_d$MH.invokeExact(indexPointer);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_Index_free$MH =
      getHandle("faiss_Index_free", FunctionDescriptor.ofVoid(ADDRESS));

  void faiss_Index_free(MemorySegment indexPointer) {
    try {
      faiss_Index_free$MH.invokeExact(indexPointer);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_Index_is_trained$MH =
      getHandle("faiss_Index_is_trained", FunctionDescriptor.of(JAVA_INT, ADDRESS));

  int faiss_Index_is_trained(MemorySegment indexPointer) {
    try {
      return (int) faiss_Index_is_trained$MH.invokeExact(indexPointer);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_Index_search$MH =
      getHandle(
          "faiss_Index_search",
          FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));

  int faiss_Index_search(
      MemorySegment indexPointer,
      int n,
      MemorySegment queries,
      int k,
      MemorySegment distances,
      MemorySegment ids) {
    try {
      return (int)
          faiss_Index_search$MH.invokeExact(indexPointer, n, queries, k, distances, ids);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_Index_search_with_params$MH =
      getHandle(
          "faiss_Index_search_with_params",
          FunctionDescriptor.of(
              JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS));

  int faiss_Index_search_with_params(
      MemorySegment indexPointer,
      int n,
      MemorySegment queries,
      int k,
      MemorySegment searchParams,
      MemorySegment distances,
      MemorySegment ids) {
    try {
      return (int)
          faiss_Index_search_with_params$MH.invokeExact(
              indexPointer, n, queries, k, searchParams, distances, ids);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_Index_train$MH =
      getHandle("faiss_Index_train", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));

  int faiss_Index_train(MemorySegment indexPointer, int n, MemorySegment docs) {
    try {
      return (int) faiss_Index_train$MH.invokeExact(indexPointer, n, docs);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_ParameterSpace_free$MH =
      getHandle("faiss_ParameterSpace_free", FunctionDescriptor.ofVoid(ADDRESS));

  void faiss_ParameterSpace_free(MemorySegment parameterSpacePointer) {
    try {
      faiss_ParameterSpace_free$MH.invokeExact(parameterSpacePointer);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_ParameterSpace_new$MH =
      getHandle("faiss_ParameterSpace_new", FunctionDescriptor.of(JAVA_INT, ADDRESS));

  int faiss_ParameterSpace_new(MemorySegment pointer) {
    try {
      return (int) faiss_ParameterSpace_new$MH.invokeExact(pointer);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_ParameterSpace_set_index_parameters$MH =
      getHandle(
          "faiss_ParameterSpace_set_index_parameters",
          FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));

  int faiss_ParameterSpace_set_index_parameters(
      MemorySegment parameterSpacePointer, MemorySegment indexPointer, MemorySegment params) {
    try {
      return (int)
          faiss_ParameterSpace_set_index_parameters$MH.invokeExact(
              parameterSpacePointer, indexPointer, params);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_SearchParameters_free$MH =
      getHandle("faiss_SearchParameters_free", FunctionDescriptor.ofVoid(ADDRESS));

  void faiss_SearchParameters_free(MemorySegment searchParametersPointer) {
    try {
      faiss_SearchParameters_free$MH.invokeExact(searchParametersPointer);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_SearchParameters_new$MH =
      getHandle("faiss_SearchParameters_new", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));

  int faiss_SearchParameters_new(MemorySegment pointer, MemorySegment idSelectorPointer) {
    try {
      return (int) faiss_SearchParameters_new$MH.invokeExact(pointer, idSelectorPointer);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_fvec_renorm_L2$MH =
      getHandle("faiss_fvec_renorm_L2", FunctionDescriptor.ofVoid(JAVA_INT, JAVA_INT, ADDRESS));

  void faiss_fvec_renorm_L2(int d, int n, MemorySegment docs) {
    try {
      faiss_fvec_renorm_L2$MH.invokeExact(d, n, docs);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_get_last_error$MH =
      getHandle("faiss_get_last_error", FunctionDescriptor.of(ADDRESS));

  MemorySegment faiss_get_last_error() {
    try {
      return (MemorySegment) faiss_get_last_error$MH.invokeExact();
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_index_factory$MH =
      getHandle("faiss_index_factory", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT));

  int faiss_index_factory(
      MemorySegment pointer, int dimension, MemorySegment description, int metric) {
    try {
      return (int) faiss_index_factory$MH.invokeExact(pointer, dimension, description, metric);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_read_index_custom$MH =
      getHandle("faiss_read_index_custom", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));

  int faiss_read_index_custom(
      MemorySegment customIOReaderPointer, int ioFlags, MemorySegment pointer) {
    try {
      return (int)
          faiss_read_index_custom$MH.invokeExact(customIOReaderPointer, ioFlags, pointer);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  private final MethodHandle faiss_write_index_custom$MH =
      getHandle("faiss_write_index_custom", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT));

  int faiss_write_index_custom(
      MemorySegment indexPointer, MemorySegment customIOWriterPointer, int ioFlags) {
    try {
      return (int)
          faiss_write_index_custom$MH.invokeExact(indexPointer, customIOWriterPointer, ioFlags);
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable t) {
      throw new AssertionError(t);
    }
  }

  static class Exception {
    private static final MethodHandle FAISS_GET_LAST_ERROR_HANDLE =
        getHandle("faiss_get_last_error", FunctionDescriptor.of(ADDRESS));

    static void handleException(int code) {
      if (code != 0) {
        String error;
        try {
          error =
              ((MemorySegment) FAISS_GET_LAST_ERROR_HANDLE.invokeExact())
                  .reinterpret(Long.MAX_VALUE)
                  .getString(0);
        } catch (Throwable t) {
          throw new AssertionError("Error getting exception details from Faiss", t);
        }
        throw new RuntimeException(String.format(Locale.ROOT, "Faiss error (%d): %s", code, error));
      }
    }
  }
}

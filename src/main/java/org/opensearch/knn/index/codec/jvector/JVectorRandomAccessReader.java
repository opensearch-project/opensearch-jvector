/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

@Log4j2
public class JVectorRandomAccessReader implements RandomAccessReader {
    private final byte[] internalBuffer = new byte[Long.BYTES];
    private final IndexInput indexInputDelegate;
    private volatile boolean closed = false;

    public JVectorRandomAccessReader(IndexInput indexInputDelegate) {
        this.indexInputDelegate = indexInputDelegate;
    }

    @Override
    public void seek(long offset) throws IOException {
        indexInputDelegate.seek(offset);
    }

    @Override
    public long getPosition() throws IOException {
        return indexInputDelegate.getFilePointer();
    }

    @Override
    public int readInt() throws IOException {
        indexInputDelegate.readBytes(internalBuffer, 0, Integer.BYTES);
        // Replace ByteArray.getInt with ByteBuffer implementation
        return ByteBuffer.wrap(internalBuffer, 0, Integer.BYTES).order(ByteOrder.BIG_ENDIAN).getInt();
    }

    @Override
    public float readFloat() throws IOException {
        return Float.intBitsToFloat(readInt());
    }

    @Override
    public long readLong() throws IOException {
        indexInputDelegate.readBytes(internalBuffer, 0, Long.BYTES);
        return ByteBuffer.wrap(internalBuffer, 0, Long.BYTES).order(ByteOrder.BIG_ENDIAN).getLong();
    }

    @Override
    public void readFully(byte[] bytes) throws IOException {
        indexInputDelegate.readBytes(bytes, 0, bytes.length);
    }

    @Override
    public void readFully(ByteBuffer buffer) throws IOException {
        // validate that the requested bytes actually exist ----
        long remainingInFile = indexInputDelegate.length() - indexInputDelegate.getFilePointer();
        if (buffer.remaining() > remainingInFile) {
            throw new EOFException(
                    "Requested " + buffer.remaining() + " bytes but only " + remainingInFile + " available");
        }

        // Heap buffers with a backing array can be filled in one call ----
        if (buffer.hasArray()) {
            int off   = buffer.arrayOffset() + buffer.position();
            int len   = buffer.remaining();
            indexInputDelegate.readBytes(buffer.array(), off, len);
            buffer.position(buffer.limit());           // advance fully
            return;
        }

        // Direct / non-array buffers: copy in reasonable chunks ----
        while (buffer.hasRemaining()) {
            final int bytesToRead = Math.min(buffer.remaining(), Long.BYTES);
            indexInputDelegate.readBytes(this.internalBuffer, 0, bytesToRead);
            buffer.put(this.internalBuffer, 0, bytesToRead);
        }
    }

    @Override
    public void readFully(long[] vector) throws IOException {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = readLong();
        }
    }

    @Override
    public void read(int[] ints, int offset, int count) throws IOException {
        for (int i = 0; i < count; i++) {
            ints[offset + i] = readInt();
        }
    }

    @Override
    public void read(float[] floats, int offset, int count) throws IOException {
        // Note that we are not using the readFloats method from IndexInput because it does not support the endianess correctly as is
        // written by the jvector writer
        for (int i = 0; i < count; i++) {
            floats[offset + i] = readFloat();
        }
    }

    @Override
    public void close() throws IOException {
        log.info("Attempting to close JVectorRandomAccessReader for file {}", indexInputDelegate);
        boolean success = false;
        if (!closed) {
            try {
                indexInputDelegate.close();
                log.info("Successfully closed JVectorRandomAccessReader for file {}", indexInputDelegate);
                success = true;
            } catch (Exception e) {
                log.error("Error closing JVectorRandomAccessReader for file {}", indexInputDelegate, e);
            } finally {
                if (!closed && !success) {
                    IOUtils.closeWhileHandlingException(this::close);
                    log.info("Closed JVectorRandomAccessReader after handling exception for file {}", indexInputDelegate);
                }
                closed = true;
            }
        } else {
            log.info("JVectorRandomAccessReader already closed for file {}", indexInputDelegate);
        }
    }

    public static class Supplier implements ReaderSupplier {
        private final Directory directory;
        private final String fileName;
        private final IOContext context;
        private final AtomicInteger readerCount = new AtomicInteger(0);
        private final ConcurrentHashMap<Integer, RandomAccessReader> readers = new ConcurrentHashMap<>();

        public Supplier(Directory directory, String fileName, IOContext context) {
            this.directory = directory;
            this.fileName = fileName;
            this.context = context;
        }

        @Override
        public RandomAccessReader get() throws IOException {
            synchronized (directory) {
                var reader = new JVectorRandomAccessReader(directory.openInput(fileName, context));
                readers.put(readerCount.getAndIncrement(), reader);
                return reader;
            }
        }

        @Override
        public void close() throws IOException {
            for (RandomAccessReader reader : readers.values()) {
                IOUtils.close(reader::close);
            }
        }
    }
}

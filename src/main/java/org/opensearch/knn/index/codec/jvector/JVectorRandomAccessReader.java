/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

@Log4j2
public class JVectorRandomAccessReader implements RandomAccessReader {
    private final byte[] internalBuffer = new byte[Long.BYTES];
    private final IndexInput indexInputDelegate;
    private final boolean ownsDelegate;
    private volatile boolean closed = false;

    public JVectorRandomAccessReader(IndexInput indexInputDelegate) {
        this(indexInputDelegate, false);
    }

    public JVectorRandomAccessReader(IndexInput indexInputDelegate, boolean ownsDelegate) {
        this.indexInputDelegate = indexInputDelegate;
        this.ownsDelegate = ownsDelegate;
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
        return indexInputDelegate.readInt();
    }

    @Override
    public float readFloat() throws IOException {
        return Float.intBitsToFloat(indexInputDelegate.readInt());
    }

    // TODO: bring back to override when upgrading jVector again
    // @Override
    public long readLong() throws IOException {
        return indexInputDelegate.readLong();
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
            throw new EOFException("Requested " + buffer.remaining() + " bytes but only " + remainingInFile + " available");
        }

        // Heap buffers with a backing array can be filled in one call ----
        if (buffer.hasArray()) {
            int off = buffer.arrayOffset() + buffer.position();
            int len = buffer.remaining();
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
        final ByteBuffer byteBuffer = ByteBuffer.allocate(Float.BYTES * count);
        indexInputDelegate.readBytes(byteBuffer.array(), offset, Float.BYTES * count);
        FloatBuffer buffer = byteBuffer.asFloatBuffer();
        buffer.get(floats, offset, count);
    }

    @Override
    public void close() throws IOException {
        if (this.closed == true) {
            log.debug("JVectorRandomAccessReader already closed for file: {}", indexInputDelegate);
            return;
        }
        log.debug("Closing JVectorRandomAccessReader for file: {}", indexInputDelegate);
        this.closed = true;

        // Only close the delegate when this reader owns it.
        // In other case, no need to really close the index input delegate since it is a clone
        if (ownsDelegate) {
            IOUtils.closeWhileHandlingException(indexInputDelegate);
        }

        log.debug("Closed JVectorRandomAccessReader for file: {}", indexInputDelegate);
    }

    @Override
    public long length() throws IOException {
        return indexInputDelegate.length();
    }

    /**
     * Supplies readers which are actually slices of the original IndexInput.
     * We will vend out slices in order for us to easily find the footer of the jVector graph index.
     * This is useful because our logic that reads the graph that the footer is always at {@link IndexInput#length()} of the slice.
     * Which is how {@link io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex#load(ReaderSupplier, long)} is working behind the scenes.
     * The header offset, on the other hand, is flexible because we can provide it as a parameter to {@link io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex#load(ReaderSupplier, long)}
     */
    public static class Supplier implements ReaderSupplier {
        private final IndexInput currentInput;
        private final long sliceStartOffset;
        private final long sliceLength;

        public Supplier(IndexInput indexInput) throws IOException {
            this(indexInput, indexInput.getFilePointer(), indexInput.length() - indexInput.getFilePointer());
        }

        public Supplier(IndexInput indexInput, long sliceStartOffset, long sliceLength) throws IOException {
            this.currentInput = indexInput;
            this.sliceStartOffset = sliceStartOffset;
            this.sliceLength = sliceLength;
        }

        @Override
        public RandomAccessReader get() throws IOException {
            synchronized (this) {
                final IndexInput input = currentInput.slice("Input Slice for the jVector graph or PQ", sliceStartOffset, sliceLength)
                    .clone();
                return new JVectorRandomAccessReader(input);
            }
        }

        @Override
        public void close() throws IOException {
            IOUtils.closeWhileHandlingException(currentInput);
        }
    }

    /**
     * Supplies readers that can support concurrent usages.
     * Every search request calls index.getView() which calls get().
     * It opens a fresh IndexInput per get() call, giving each concurrent search
     * its own independent file position cursor with no shared mutable state.
     * The caller is responsible for closing each reader. Close method
     * on this supplier is a no-op because no shared resource is held.
     */
    public static class IndependentSupplier implements ReaderSupplier {
        private final Directory directory;
        private final String fileName;
        private final IOContext context;
        private final long sliceLength;

        public IndependentSupplier(Directory directory, String fileName, IOContext context, long sliceLength) {
            this.directory = directory;
            this.fileName = fileName;
            this.context = context;
            this.sliceLength = sliceLength;
        }

        @Override
        public RandomAccessReader get() throws IOException {
            final IndexInput input = directory.openInput(fileName, context).slice("jVector graph index slice", 0, sliceLength);
            // Setting `ownsDelegate=true` this reader is only responsible for closing this IndexInputd
            return new JVectorRandomAccessReader(input, true);
        }

        @Override
        public void close() {
            // Nothing to close — this supplier holds no shared resource.
        }
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.apache.lucene.store.IndexInput;
import org.junit.Assert;
import org.junit.Test;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class JVectorRandomAccessReaderTests {

    /** Minimal in-memory IndexInput for test purposes. */
    private static final class ByteArrayIndexInput extends IndexInput {
        private final byte[] data;
        private long pos = 0;

        ByteArrayIndexInput(byte[] data) {
            super("ByteArrayIndexInput");
            this.data = data;
        }

        @Override
        public void close() {/* nothing to close */}

        @Override
        public long length() {
            return data.length;
        }

        @Override
        public long getFilePointer() {
            return pos;
        }

        @Override
        public void seek(long newPos) {
            this.pos = newPos;
        }

        @Override
        public byte readByte() throws EOFException {
            if (pos >= data.length) throw new EOFException();
            return data[(int) pos++];
        }

        @Override
        public void readBytes(byte[] b, int offset, int len) throws EOFException {
            if (pos + len > data.length) throw new EOFException();
            System.arraycopy(data, (int) pos, b, offset, len);
            pos += len;
        }

        @Override
        public IndexInput slice(String desc, long offset, long length) {
            throw new UnsupportedOperationException("slice() not needed for these tests");
        }

        @Override
        public IndexInput clone() {
            ByteArrayIndexInput clone = new ByteArrayIndexInput(data);
            clone.pos = this.pos;
            return clone;
        }
    }

    /* ------------------------------------------------------------------ */

    /**
     * Tests the behavior of the {@code readFully(ByteBuffer)} method in the
     * {@code JVectorRandomAccessReader} class when filling a heap {@code ByteBuffer}.
     *
     * This method verifies the following:
     * - A heap {@code ByteBuffer} with a non-zero initial position is filled with
     *   data from the source while respecting the buffer's position and limit.
     * - The buffer's position is advanced correctly such that no space remains in
     *   the buffer after the read operation.
     * - The contents of the buffer match the expected values from the source data,
     *   ensuring data integrity during the read process.
     * - The additional untouched portion of the buffer (before the initial position)
     *   remains unchanged.
     *
     * @throws IOException If an I/O error occurs during the read operation.
     */
    @Test
    public void readFully_fillsHeapBuffer() throws IOException {
        byte[] src = new byte[32];
        for (int i = 0; i < src.length; i++)
            src[i] = (byte) i;

        JVectorRandomAccessReader reader = new JVectorRandomAccessReader(new ByteArrayIndexInput(src));

        // Create a heap buffer with non-zero position to exercise offset math
        ByteBuffer dst = ByteBuffer.allocate(16);
        dst.position(4);          // leave first four bytes untouched
        dst.limit(16);            // want to read 12 bytes

        reader.readFully(dst);

        Assert.assertFalse("Buffer should be full", dst.hasRemaining());
        Assert.assertEquals("Exactly 12 bytes expected", 12, dst.position() - 4);

        // Validate contents 0..11 from source array
        dst.rewind();
        byte[] actual = new byte[16];
        dst.get(actual);
        byte[] expected = new byte[16];
        System.arraycopy(src, 0, expected, 4, 12);
        Assert.assertArrayEquals(expected, actual);
    }

    /**
     * Tests the behavior of the {@code readFully(ByteBuffer)} method in the
     * {@code JVectorRandomAccessReader} class when filling a direct {@code ByteBuffer}.
     *
     * The method verifies the following:
     * - The direct {@code ByteBuffer} is filled with the data from the source.
     * - The buffer's position is advanced correctly such that it has no remaining space
     *   after the read operation.
     * - The content of the buffer matches the expected values from the source data,
     *   ensuring data integrity during the reading process.
     */
    @Test
    public void readFully_fillsDirectBuffer() throws IOException {
        byte[] src = new byte[64];
        for (int i = 0; i < src.length; i++)
            src[i] = (byte) (i * 2);

        JVectorRandomAccessReader reader = new JVectorRandomAccessReader(new ByteArrayIndexInput(src));

        ByteBuffer dst = ByteBuffer.allocateDirect(40);   // triggers chunk-copy path
        reader.readFully(dst);

        Assert.assertFalse(dst.hasRemaining());
        dst.flip();
        for (int i = 0; i < 40; i++) {
            Assert.assertEquals("Mismatch at index " + i, src[i], dst.get());
        }
    }

    /**
     * Test that readFully throws EOFException when there is insufficient data.
     */
    @Test
    public void readFully_throwsEOFException_whenInsufficientData() {
        byte[] src = new byte[8];
        JVectorRandomAccessReader reader = new JVectorRandomAccessReader(new ByteArrayIndexInput(src));

        ByteBuffer dst = ByteBuffer.allocate(16);
        Assert.assertThrows(EOFException.class, () -> reader.readFully(dst));
    }

    private static class CapturingIndexInput extends IndexInput {
        final List<long[]> calls = new ArrayList<>();
        private final long size;

        CapturingIndexInput(long size) {
            super("CapturingIndexInput");
            this.size = size;
        }

        @Override
        public void close() {}

        @Override
        public long getFilePointer() {
            return 0L;
        }

        @Override
        public void seek(long pos) {}

        @Override
        public long length() {
            return size;
        }

        @Override
        public byte readByte() {
            return 0;
        }

        @Override
        public void readBytes(byte[] b, int off, int len) {}

        @Override
        public IndexInput clone() {
            return this;
        }

        @Override
        public IndexInput slice(String d, long o, long l) {
            return new CapturingIndexInput(l);
        }

        @Override
        public void prefetch(long offset, long length) throws IOException {
            calls.add(new long[] { offset, length });
        }
    }

    /** Supplier.prefetch forwards offset and length to IndexInput.prefetch. */
    @Test
    public void supplierPrefetch_forwardsToIndexInput() throws IOException {
        CapturingIndexInput input = new CapturingIndexInput(1024);
        JVectorRandomAccessReader.Supplier supplier = new JVectorRandomAccessReader.Supplier(input, 0L, 1024L);

        supplier.prefetch(100L, 64L);

        Assert.assertEquals(1, input.calls.size());
        Assert.assertArrayEquals(new long[] { 100L, 64L }, input.calls.get(0));
    }

    /** Multiple consecutive calls are each forwarded independently. */
    @Test
    public void supplierPrefetch_multipleCallsEachForwarded() throws IOException {
        CapturingIndexInput input = new CapturingIndexInput(4096);
        JVectorRandomAccessReader.Supplier supplier = new JVectorRandomAccessReader.Supplier(input, 0L, 4096L);

        supplier.prefetch(0L, 128L);
        supplier.prefetch(512L, 256L);
        supplier.prefetch(1024L, 512L);

        Assert.assertEquals(3, input.calls.size());
        Assert.assertArrayEquals(new long[] { 0L, 128L }, input.calls.get(0));
        Assert.assertArrayEquals(new long[] { 512L, 256L }, input.calls.get(1));
        Assert.assertArrayEquals(new long[] { 1024L, 512L }, input.calls.get(2));
    }

    /** IOException thrown by IndexInput.prefetch must be swallowed silently. */
    @Test
    public void supplierPrefetch_ioExceptionSwallowed() throws IOException {
        CapturingIndexInput input = new CapturingIndexInput(1024) {
            @Override
            public void prefetch(long offset, long length) throws IOException {
                throw new IOException("simulated madvise failure");
            }
        };
        JVectorRandomAccessReader.Supplier supplier = new JVectorRandomAccessReader.Supplier(input, 0L, 1024L);

        // must not propagate
        supplier.prefetch(0L, 64L);
    }
}

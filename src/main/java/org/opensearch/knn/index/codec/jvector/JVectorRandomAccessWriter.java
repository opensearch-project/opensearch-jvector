/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.disk.RandomAccessWriter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * JVectorRandomAccessWriter is a wrapper around IndexOutput that implements RandomAccessWriter.
 * Note: This is not thread safe!
 */
@Log4j2
public class JVectorRandomAccessWriter implements RandomAccessWriter {
    private final byte[] writeBuffer = new byte[Long.BYTES]; // used to store temporary bytes conversion before writing
    private final IndexOutput indexOutputDelegate;

    public JVectorRandomAccessWriter(IndexOutput indexOutputDelegate) {
        this.indexOutputDelegate = indexOutputDelegate;
    }

    @Override
    public void seek(long position) throws IOException {
        log.info("Pretending to be seeking to position {} in JVectorRandomAccessWriter, in practice it is not seeking anywhere.", position);
    }

    @Override
    public long position() throws IOException {
        return indexOutputDelegate.getFilePointer();
    }

    @Override
    public void flush() throws IOException {
        log.info(
            "Pretending to be flushing JVectorRandomAccessWriter, in practice it is not flushing anything. It will be flushed when the IndexOutput is closed"
        );
    }

    @Override
    public long checksum(long startOffset, long endOffset) throws IOException {
        log.info(
            "Pretending to be computing checksum for range [{}, {}) in JVectorRandomAccessWriter, in practice it is not computing anything.",
            startOffset,
            endOffset
        );
        return 0;
    }

    @Override
    public void close() throws IOException {
        indexOutputDelegate.close();
    }

    @Override
    public void write(int b) throws IOException {
        indexOutputDelegate.writeByte((byte) b);
    }

    @Override
    public void write(byte[] b) throws IOException {
        indexOutputDelegate.writeBytes(b, 0, b.length);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
        indexOutputDelegate.writeBytes(b, off, len);
    }

    @Override
    public void writeBoolean(boolean v) throws IOException {
        indexOutputDelegate.writeByte((byte) (v ? 1 : 0));
    }

    @Override
    public void writeByte(int v) throws IOException {
        indexOutputDelegate.writeByte((byte) v);
    }

    @Override
    public void writeShort(int v) throws IOException {
        indexOutputDelegate.writeShort((short) v);
    }

    @Override
    public void writeChar(int v) throws IOException {
        throw new UnsupportedOperationException("JVectorRandomAccessWriter does not support writing chars");
    }

    @Override
    public void writeInt(int v) throws IOException {
        indexOutputDelegate.writeInt(v);
    }

    @Override
    public void writeLong(long v) throws IOException {
        indexOutputDelegate.writeLong(v);
    }

    @Override
    public void writeFloat(float v) throws IOException {
        indexOutputDelegate.writeInt(Float.floatToIntBits(v));
    }

    @Override
    public void writeDouble(double v) throws IOException {
        writeLong(Double.doubleToLongBits(v));
    }

    @Override
    public void writeBytes(String s) throws IOException {
        throw new UnsupportedOperationException("JVectorRandomAccessWriter does not support writing bytes");
    }

    @Override
    public void writeChars(String s) throws IOException {
        throw new UnsupportedOperationException("JVectorRandomAccessWriter does not support writing chars");
    }

    @Override
    public void writeUTF(String s) throws IOException {
        throw new UnsupportedOperationException("JVectorRandomAccessWriter does not support writing UTF strings");
    }
}

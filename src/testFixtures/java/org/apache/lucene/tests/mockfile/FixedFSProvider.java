/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.apache.lucene.tests.mockfile;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.attribute.FileAttribute;
import java.util.Set;

/**
 * A fixed file system provider for the LuceneTestCases.
 * Those otherwise break due to the failover to the default implementation of {@link FileChannel#map} that is not implemented.
 *
 * This FileSystem provider is to be wrapped up around every LuceneTestCase generated Path in the following manner
 * Path luceneGeneratedPath = createTempDir();
 * Path wrappedPath = new FixedFSProvider(luceneGeneratedPath.getFileSystem()).wrapPath(luceneGeneratedPath);
 * this can later be used within the tests with no concerns regarding the {@link FileChannel#map} functionality to throw an UnsupportedOperationException exception.
 */
public class FixedFSProvider extends FilterFileSystemProvider {
    /**
     * Create a new instance, wrapping {@code delegate}.
     *
     * @param delegate
     */
    public FixedFSProvider(FileSystem delegate) {
        super("fixedFs://", delegate);
    }

    @Override
    public FileChannel newFileChannel(Path path, Set<? extends OpenOption> options, FileAttribute<?>... attrs) throws IOException {
        while (path instanceof FilterPath) {
            path = ((FilterPath) path).getDelegate();
        }
        return new FilterFileChannel(FileSystems.getDefault().provider().newFileChannel(path, options, attrs)) {
            @Override
            public MemorySegment map(MapMode mode, long offset, long size, Arena arena) throws IOException {
                return delegate.map(mode, offset, size, arena);
            }
        };
    }
}

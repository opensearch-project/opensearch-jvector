/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

/**
 * Benchmark to compare the graph index construction performance with jVector in various configurations and Lucene
 * The benchmark generates random vectors and indexes them using JVector and Lucene codecs.
 * Note: This benchmark is not meant to reproduce the already existing benchmarks of either Lucene or JVector.
 * But rather it is more meant as a qualitative analysis of the relative performance of the codecs in the plugin for certain scenarios.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Fork(1)
public class FormatBenchmarkConstructionWithRandomVectors {
    private static final Logger log = LogManager.getLogger(FormatBenchmarkConstructionWithRandomVectors.class);
    private static final String JVECTOR_NOT_QUANTIZED = "jvector_not_quantized";
    private static final String JVECTOR_QUANTIZED = "jvector_quantized";
    private static final String LUCENE101 = "Lucene101";
    private static final String FIELD_NAME = "vector_field";
    private static final VectorSimilarityFunction SIMILARITY_FUNCTION = VectorSimilarityFunction.EUCLIDEAN;
    @Param({ JVECTOR_NOT_QUANTIZED, JVECTOR_QUANTIZED, LUCENE101 })  // This will run the benchmark each codec type
    private String codecType;
    @Param({ /*"1000", "10000",*/ "100000" })
    private int numDocs;
    @Param({ /*"128", "256",*/ "768", /*"1024"*/ })
    private int dimension;

    private float[][] vectors;
    private Directory directory;
    private Path indexDirectoryPath;
    private List<Document> documents;

    @Setup(Level.Invocation)
    public void setup() throws IOException {
        documents = new ArrayList<>(numDocs);
        vectors = new float[numDocs][dimension];
        log.info("Generating {} random vectors of dimension {}", numDocs, dimension);
        // Generate random vectors
        ThreadLocalRandom random = ThreadLocalRandom.current();
        for (int i = 0; i < numDocs; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = random.nextFloat();
            }
        }

        indexDirectoryPath = Files.createTempDirectory("jvector-benchmark");
        log.info("Index path: {}", indexDirectoryPath);
        directory = FSDirectory.open(indexDirectoryPath);

        for (int i = 0; i < numDocs; i++) {
            Document doc = new Document();
            doc.add(new KnnFloatVectorField(FIELD_NAME, vectors[i], SIMILARITY_FUNCTION));
            documents.add(doc);
        }
    }

    @TearDown(Level.Invocation)
    public void tearDown() throws IOException {
        directory.close();
        // Cleanup previously created index directory
        Files.walk(indexDirectoryPath)
            .sorted((path1, path2) -> path2.compareTo(path1)) // Reverse order to delete files before directories
            .forEach(path -> {
                try {
                    Files.delete(path);
                } catch (IOException e) {
                    throw new UncheckedIOException("Failed to delete " + path, e);
                }
            });
        documents.clear();
    }

    @Benchmark
    public void benchmarkSearch(Blackhole blackhole) throws IOException {
        IndexWriterConfig indexWriterConfig = new IndexWriterConfig();
        indexWriterConfig.setCodec(BenchmarkCommon.getCodec(codecType));
        indexWriterConfig.setUseCompoundFile(true);
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(true));
        try (IndexWriter writer = new IndexWriter(directory, indexWriterConfig)) {
            writer.addDocuments(documents);
            writer.commit();
            log.info("Flushing docs to make them discoverable on the file system and force merging all segments to get a single segment");
            writer.forceMerge(1);
        }
        blackhole.consume(documents);
    }
}

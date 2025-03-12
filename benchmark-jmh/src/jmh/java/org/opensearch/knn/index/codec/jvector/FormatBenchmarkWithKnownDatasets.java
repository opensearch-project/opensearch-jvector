/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec.jvector;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSLockFactory;
import org.apache.lucene.store.NIOFSDirectory;
import org.openjdk.jmh.annotations.*;
import org.opensearch.knn.index.codec.jvector.datasets.DataSet;
import org.opensearch.knn.index.codec.jvector.datasets.DownloadHelper;
import org.opensearch.knn.index.codec.jvector.datasets.Hdf5Loader;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.opensearch.knn.index.codec.jvector.BenchmarkCommon.*;

/************************************************************
 * This benchmark tests the performance of the JVector and Lucene codecs in the plugin
 * with known datasets.
 * Note: Keep in mind that this benchmark is not meant to reproduce the already existing benchmarks of either Lucene or JVector.
 * But rather it is more meant as a qualitative analysis of the relative performance of the codecs in the plugin for certain scenarios.
 ************************************************************/
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Fork(1)
public class FormatBenchmarkWithKnownDatasets {
    Logger log = LogManager.getLogger(FormatBenchmarkWithKnownDatasets.class);
    // large embeddings calculated by Neighborhood Watch. 100k files by default; 1M also available
    private static final List<String> LARGE_DATASETS = List.of(
        "ada002-100k",
        "cohere-english-v3-100k",
        "openai-v3-small-100k",
        "nv-qa-v4-100k",
        "colbert-1M",
        "gecko-100k"
    );

    // smaller vectors from ann-benchmarks
    private static final List<String> SMALL_DATASETS = List.of(
        // large files not yet supported
        // "hdf5/deep-image-96-angular.hdf5",
        // "hdf5/gist-960-euclidean.hdf5",
        "glove-25-angular.hdf5",
        "glove-50-angular.hdf5",
        "lastfm-64-dot.hdf5",
        "glove-100-angular.hdf5",
        "glove-200-angular.hdf5",
        "nytimes-256-angular.hdf5",
        "sift-128-euclidean.hdf5"
    );

    @Param({ "sift-128-euclidean.hdf5", /*"nytimes-256-angular.hdf5", "ada002-100k"*/ })
    private String datasetName;
    @Param({ JVECTOR_NOT_QUANTIZED/*, JVECTOR_QUANTIZED*/, LUCENE101 })  // This will run the benchmark each codec type
    private String codecType;
    private DataSet dataset;
    private static final int K = 100;

    private Directory directory;
    private DirectoryReader directoryReader;
    private Path indexDirectoryPath;
    private IndexSearcher searcher;
    private float[] queryVector;
    private float expectedMinScoreInTopK;
    private VectorSimilarityFunction vectorSimilarityFunction;
    private double totalRecall = 0.0;
    private int recallCount = 0;

    @Setup
    public void setup() throws IOException {
        // Download datasets
        if (LARGE_DATASETS.contains(datasetName)) {
            var mfd = DownloadHelper.maybeDownloadFvecs(datasetName);
            try {
                dataset = mfd.load();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        } else if (SMALL_DATASETS.contains(datasetName)) {
            DownloadHelper.maybeDownloadHdf5(datasetName);
            dataset = Hdf5Loader.load(datasetName);
        } else {
            throw new IllegalArgumentException("Unknown dataset: " + datasetName);
        }

        final Path indexPath = Files.createTempDirectory("jvector-benchmark");
        log.info("Index path: {}", indexPath);
        directory = new NIOFSDirectory(indexPath, FSLockFactory.getDefault());

        // Create index with JVectorFormat
        IndexWriterConfig indexWriterConfig = new IndexWriterConfig();
        indexWriterConfig.setCodec(BenchmarkCommon.getCodec(codecType));
        indexWriterConfig.setUseCompoundFile(true);
        indexWriterConfig.setMergePolicy(new ForceMergesOnlyMergePolicy(true));

        float[][] vectors = new float[dataset.baseVectors.size()][dataset.getDimension()];
        for (int i = 0; i < dataset.baseVectors.size(); i++) {
            vectors[i] = (float[]) dataset.baseVectors.get(i).get();
        }

        // Convert from jVector similarity function to Lucene similarity function
        vectorSimilarityFunction = switch (dataset.similarityFunction) {
            case COSINE -> VectorSimilarityFunction.COSINE;
            case DOT_PRODUCT -> VectorSimilarityFunction.DOT_PRODUCT;
            case EUCLIDEAN -> VectorSimilarityFunction.EUCLIDEAN;
            default -> throw new IllegalStateException("Unexpected similarity function: " + dataset.similarityFunction);
        };
        log.info("Using similarity function: {}", vectorSimilarityFunction);

        try (IndexWriter writer = new IndexWriter(directory, indexWriterConfig)) {
            for (int i = 0; i < vectors.length; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD_NAME, vectors[i], vectorSimilarityFunction));
                writer.addDocument(doc);
            }
            writer.commit();
            log.info("Flushing docs to make them discoverable on the file system and force merging all segments to get a single segment");
            writer.forceMerge(1);
        }
        directoryReader = DirectoryReader.open(directory);
        searcher = new IndexSearcher(directoryReader);
        queryVector = (float[]) dataset.queryVectors.getFirst().get();
        expectedMinScoreInTopK = findExpectedKthMaxScore(queryVector, vectors, vectorSimilarityFunction, K);
    }

    // Print average recall after each iteration
    @TearDown(Level.Iteration)
    public void printIterationStats() {
        log.info("Average recall: {}", totalRecall / recallCount);
    }

    @TearDown(Level.Trial)
    public void printFinalStats() {
        log.info("=== Benchmark Results ===");
        log.info("Total Iterations: {}", recallCount);
        log.info("Average Recall: {}", totalRecall / recallCount);
        log.info("=====================");
    }

    @TearDown
    public void tearDown() throws IOException {
        directoryReader.close();
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
    }

    @Benchmark
    public RecallResult benchmarkSearch() throws IOException {
        KnnFloatVectorQuery query = new KnnFloatVectorQuery(FIELD_NAME, queryVector, K);
        TopDocs topDocs = searcher.search(query, K);

        // Calculate recall
        float recall = calculateRecall(topDocs, expectedMinScoreInTopK);
        totalRecall += recall;
        recallCount++;
        return new RecallResult(recall);
    }
}

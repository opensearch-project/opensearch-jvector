/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import static org.opensearch.knn.index.engine.CommonTestUtils.getCodec;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.IntField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.ThreadLeakFiltersForTests;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;

import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Builder.Default;
import lombok.Singular;

/**
 * Tests for the FusedPQ write, merge, and search paths.
 *
 * <p>FusedPQ stores the PQ codes used for graph traversal inline per adjacency-list entry
 * ({@link FeatureId#FUSED_PQ}) instead of appending them as a separate blob that is loaded
 * into heap at segment open. Full-precision vectors are still kept inline for reranking
 * ({@link FeatureId#INLINE_VECTORS}). It is an on-disk <em>layout</em> flag
 * ({@code advanced.fused_pq_enabled}), orthogonal to the quantization type.
 *
 * <p>Coverage:
 * <ul>
 *   <li>Single-segment flush with FusedPQ enabled — recall + metadata flags</li>
 *   <li>Fallback to full precision when the vector count is below the batch threshold</li>
 *   <li>Two-segment and multi-segment merges (with deletions) under FusedPQ</li>
 *   <li>Multi-phase progressive merge once the vector count crosses the threshold</li>
 *   <li>All three similarity functions (EUCLIDEAN, COSINE, DOT_PRODUCT)</li>
 *   <li>Concurrent search over a FusedPQ index</li>
 *   <li>Merge with leading-segment merge disabled</li>
 * </ul>
 *
 * FusedPQ uses the same codebook and codes as the separate-blob PQ path, so recall targets
 * match the PQ equivalents.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "")
public class JVectorFusedPQTests extends LuceneTestCase {

    static final int RANDOM_SEED = 42;
    static final String ID_FIELD = "doc_id";
    static final String VECTOR_FIELD = "vectors";

    @AllArgsConstructor
    static class DeletionRange {
        int start;
        int end;
    }

    @Builder
    static class MergeTestRound {
        @Default
        List<Integer> segmentSizes = List.of();
        @Default
        List<DeletionRange> deletionRanges = List.of();
    }

    @Builder
    static class MergeTestScenario {
        @Singular
        List<MergeTestRound> rounds;
        @Default
        int minFusedPqThreshold = 1; // always apply FusedPQ by default
        @Default
        int dimension = 128;
        @Default
        int nQueries = 10;
        @Default
        int topK = 10;
        @Default
        int overqueryFactor = 10;
        @Default
        double minimumRecall = 0.7;
        @Default
        boolean leadingSegmentMergeDisabled = KNNConstants.DEFAULT_LEADING_SEGMENT_MERGE_DISABLED;
        @Default
        VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
    }

    @Rule
    public TemporaryFolder tempDir = new TemporaryFolder();
    private ForkJoinPool singleThreadGraphMergePool;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        singleThreadGraphMergePool = new ForkJoinPool(1);
    }

    @After
    public void tearDown() throws Exception {
        super.tearDown();
        singleThreadGraphMergePool.shutdown();
        if (singleThreadGraphMergePool.awaitTermination(30, TimeUnit.SECONDS) == false) {
            singleThreadGraphMergePool.shutdownNow();
        }
    }

    void runScenario(MergeTestScenario scenario) throws IOException {
        runScenarioWithPool(scenario, null);
    }

    void runScenarioWithPool(MergeTestScenario scenario, ForkJoinPool mergePool) throws IOException {
        int nBase = scenario.rounds.stream().mapToInt(r -> r.segmentSizes.stream().mapToInt(x -> x).sum()).sum();
        var baseVecs = TestUtils.randomlyGenerateStandardVectors(nBase, scenario.dimension, RANDOM_SEED);
        var queryVecs = TestUtils.randomlyGenerateStandardVectors(scenario.nQueries, scenario.dimension, RANDOM_SEED + 1);

        var liveVecs = new FixedBitSet(nBase);
        liveVecs.clear();

        IndexWriterConfig iwc = LuceneTestCase.newIndexWriterConfig();
        iwc.setUseCompoundFile(false);
        // Quantization type stays plain PQ; the trailing flag enables the FusedPQ on-disk layout.
        iwc.setCodec(
            getCodec(scenario.minFusedPqThreshold, scenario.leadingSegmentMergeDisabled, mergePool, new JVectorIndexQuantization.PQ(), true)
        );
        iwc.setMergePolicy(new ForceMergesOnlyMergePolicy(false));
        iwc.setMaxBufferedDocs(-1);

        try (var fsd = FSDirectory.open(tempDir.getRoot().toPath()); var writer = new IndexWriter(fsd, iwc)) {
            int vectorOffset = 0;
            for (int roundId = 0; roundId < scenario.rounds.size(); roundId++) {
                var round = scenario.rounds.get(roundId);

                for (int segInRound = 0; segInRound < round.segmentSizes.size(); segInRound++) {
                    var segmentSize = round.segmentSizes.get(segInRound);

                    for (int i = 0; i < segmentSize; i++) {
                        int id = vectorOffset + i;
                        Document doc = new Document();
                        doc.add(new KnnFloatVectorField(VECTOR_FIELD, baseVecs[id], scenario.similarityFunction));
                        doc.add(new IntField(ID_FIELD, id, Store.YES));
                        writer.addDocument(doc);
                        liveVecs.set(id);
                    }
                    writer.flush();
                    writer.commit();
                    vectorOffset += segmentSize;
                }

                for (var dl : round.deletionRanges) {
                    var query = IntField.newRangeQuery(ID_FIELD, dl.start, dl.end - 1);
                    writer.deleteDocuments(query);
                    liveVecs.clear(dl.start, dl.end);
                    writer.commit();
                }

                writer.forceMerge(1);
                writer.commit();

                try (var reader = DirectoryReader.open(writer)) {
                    Assert.assertTrue("Should have one segment after merge", 1 >= reader.getContext().leaves().size());
                    assertEquals(liveVecs.cardinality(), reader.numDocs());

                    var searcher = LuceneTestCase.newSearcher(reader);
                    int totalCorrect = 0;

                    for (var queryVec : queryVecs) {
                        var gt = computeGroundTruth(scenario.similarityFunction, baseVecs, liveVecs, queryVec, scenario.topK);
                        var query = new JVectorKnnFloatVectorQuery(
                            VECTOR_FIELD,
                            queryVec,
                            scenario.topK,
                            scenario.overqueryFactor,
                            0.0f,
                            0.0f
                        );
                        var results = searcher.search(query, scenario.topK).scoreDocs;

                        var pred = Arrays.stream(results).map(r -> {
                            try {
                                return reader.storedFields().document(r.doc).getField(ID_FIELD).numericValue().intValue();
                            } catch (IOException e) {
                                throw new UncheckedIOException(e);
                            }
                        }).collect(Collectors.toSet());

                        assertEquals(scenario.topK, pred.size());
                        for (var doc : pred) {
                            if (gt.contains(doc)) {
                                totalCorrect++;
                            }
                        }
                    }

                    double recall = totalCorrect / (double) (scenario.topK * queryVecs.length);
                    Assert.assertTrue(
                        String.format("FusedPQ recall too low [round %d]: got %.3f < %.3f", roundId, recall, scenario.minimumRecall),
                        recall >= scenario.minimumRecall
                    );
                }
            }
        }
    }

    Set<Integer> computeGroundTruth(VectorSimilarityFunction vsf, float[][] baseVectors, BitSet liveVectors, float[] query, int k) {
        if (liveVectors.length() == 0) {
            return Set.of();
        }
        var pq = new PriorityQueue<ScoreDoc>(k, (a, b) -> Float.compare(a.score, b.score));
        for (int i = liveVectors.nextSetBit(0); i != DocIdSetIterator.NO_MORE_DOCS; i = liveVectors.nextSetBit(i + 1)) {
            var score = vsf.compare(baseVectors[i], query);
            var sd = new ScoreDoc(i, score);
            if (pq.size() < k) {
                pq.add(sd);
            } else if (pq.peek().score < sd.score) {
                pq.poll();
                pq.add(sd);
            }
            if (i + 1 >= liveVectors.length()) break;
        }
        return pq.stream().map(sd -> sd.doc).collect(Collectors.toSet());
    }

    private JVectorReader openJVectorReader(DirectoryReader reader) throws IOException {
        LeafReaderContext ctx = reader.getContext().leaves().get(0);
        SegmentReader segReader = (SegmentReader) FilterLeafReader.unwrap(ctx.reader());
        PerFieldKnnVectorsFormat.FieldsReader outer = (PerFieldKnnVectorsFormat.FieldsReader) segReader.getVectorReader();
        return (JVectorReader) outer.getFieldReader(VECTOR_FIELD);
    }

    /**
     * Single segment, FusedPQ always active (minThreshold=1). Verifies the full
     * flush -> FusedPQ encode -> search path including recall. 512 vectors guarantees
     * a full 256-cluster codebook (the FusedPQ requirement).
     */
    @Test
    public void testFusedPQFlushRecall() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minFusedPqThreshold(1)
            .round(MergeTestRound.builder().segmentSizes(List.of(512)).build())
            .minimumRecall(0.7)
            .build();
        runScenario(scenario);
    }

    /**
     * FusedPQ enabled but minThreshold=MAX_VALUE, so quantization never triggers.
     * Recall must be perfect since full-precision vectors are used and there is no PQ layer.
     */
    @Test
    public void testFusedPQBelowThresholdFallbackToFullPrecision() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minFusedPqThreshold(Integer.MAX_VALUE)
            .round(MergeTestRound.builder().segmentSizes(List.of(200)).build())
            .minimumRecall(1.0)
            .overqueryFactor(KNNConstants.DEFAULT_OVER_QUERY_FACTOR)
            .build();
        runScenario(scenario);
    }

    /**
     * Two 300-vector segments merged into one under FusedPQ. Each flush segment gets 256
     * clusters (min(256, 300)). Verifies the merge path produces a valid FusedPQ segment.
     */
    @Test
    public void testFusedPQSimpleMerge() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minFusedPqThreshold(1)
            .round(MergeTestRound.builder().segmentSizes(List.of(300, 300)).build())
            .minimumRecall(0.7)
            .build();
        runScenarioWithPool(scenario, singleThreadGraphMergePool);
    }

    /**
     * Multiple segments with deletions merged under FusedPQ; each segment has >= 256 vectors
     * so the 256-cluster codebook requirement holds at flush time.
     */
    @Test
    public void testFusedPQMergeWithDeletions() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minFusedPqThreshold(1)
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(300, 300, 300))
                    .deletionRanges(List.of(new DeletionRange(0, 50), new DeletionRange(500, 600)))
                    .build()
            )
            .minimumRecall(0.7)
            .build();
        runScenarioWithPool(scenario, singleThreadGraphMergePool);
    }

    /**
     * Progressive threshold: the first two rounds stay below it (full precision), the third
     * crosses it and FusedPQ kicks in on merge. Individual flush segments stay below the
     * threshold so they remain full-precision until merged.
     */
    @Test
    public void testFusedPQMultiPhaseMergeProgressiveThreshold() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minFusedPqThreshold(1200)
            .round(MergeTestRound.builder().segmentSizes(List.of(100, 150, 150)).build())   // 400 < 1200
            .round(MergeTestRound.builder().segmentSizes(List.of(100, 150, 150)).build())   // 800 < 1200
            .round(MergeTestRound.builder().segmentSizes(List.of(200, 200, 200)).build())   // 1400 > 1200
            .overqueryFactor(20)
            .minimumRecall(0.7)
            .build();
        runScenarioWithPool(scenario, singleThreadGraphMergePool);
    }

    /**
     * FusedPQ always active with complex deletions across rounds. All flush segments use
     * >= 300 vectors to guarantee 256-cluster codebooks.
     */
    @Test
    public void testFusedPQMultiPhaseMergeWithDeletions() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minFusedPqThreshold(1)
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(300, 300, 300))
                    .deletionRanges(List.of(new DeletionRange(0, 30), new DeletionRange(600, 700)))
                    .build()
            )
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(300, 300))
                    .deletionRanges(List.of(new DeletionRange(900, 1000), new DeletionRange(50, 55)))
                    .build()
            )
            .round(MergeTestRound.builder().segmentSizes(List.of(300)).build())
            .overqueryFactor(20)
            .minimumRecall(0.7)
            .build();
        runScenarioWithPool(scenario, singleThreadGraphMergePool);
    }

    /** FusedPQ with COSINE similarity (exercises the cosine decoder path). */
    @Test
    public void testFusedPQCosineRecall() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minFusedPqThreshold(1)
            .round(MergeTestRound.builder().segmentSizes(List.of(512)).build())
            .similarityFunction(VectorSimilarityFunction.COSINE)
            .minimumRecall(0.7)
            .build();
        runScenario(scenario);
    }

    /** FusedPQ with DOT_PRODUCT similarity (exercises the dot-product decoder path). */
    @Test
    public void testFusedPQDotProductRecall() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minFusedPqThreshold(1)
            .round(MergeTestRound.builder().segmentSizes(List.of(512)).build())
            .similarityFunction(VectorSimilarityFunction.DOT_PRODUCT)
            .minimumRecall(0.6)
            .build();
        runScenario(scenario);
    }

    /**
     * After a FusedPQ flush the on-disk graph carries both INLINE_VECTORS (reranking) and
     * FUSED_PQ (traversal), and no separate PQ blob is loaded.
     */
    @Test
    public void testFusedPQMetadataFlags() throws IOException {
        int numVectors = 512;

        IndexWriterConfig iwc = LuceneTestCase.newIndexWriterConfig();
        iwc.setUseCompoundFile(false);
        iwc.setCodec(getCodec(1, KNNConstants.DEFAULT_LEADING_SEGMENT_MERGE_DISABLED, null, new JVectorIndexQuantization.PQ(), true));
        iwc.setMergePolicy(new ForceMergesOnlyMergePolicy(false));
        iwc.setMaxBufferedDocs(-1);

        float[][] vecs = TestUtils.randomlyGenerateStandardVectors(numVectors, 128, RANDOM_SEED);

        try (var fsd = FSDirectory.open(tempDir.getRoot().toPath()); var writer = new IndexWriter(fsd, iwc)) {
            for (int i = 0; i < numVectors; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(VECTOR_FIELD, vecs[i], VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc);
            }
            writer.flush();
            writer.commit();
            writer.forceMerge(1);
            writer.commit();

            try (var reader = DirectoryReader.open(writer)) {
                JVectorReader jvReader = openJVectorReader(reader);

                Assert.assertTrue(
                    "FusedPQ segment must not load a separate PQ blob",
                    jvReader.getProductQuantizationForField(VECTOR_FIELD).isEmpty()
                );

                var featureSet = jvReader.getOnDiskGraphIndex(VECTOR_FIELD).getFeatureSet();
                Assert.assertTrue("Graph must contain INLINE_VECTORS for reranking", featureSet.contains(FeatureId.INLINE_VECTORS));
                Assert.assertTrue("Graph must contain FUSED_PQ for traversal", featureSet.contains(FeatureId.FUSED_PQ));
            }
        }
    }

    /**
     * Concurrent searches over a FusedPQ index. Exercises the per-View lazily-read packed
     * neighbor codes under contention.
     */
    @Test
    public void testFusedPQConcurrentSearch() throws IOException, InterruptedException {
        int numVectors = 512;
        int numThreads = 4;
        int queriesPerThread = 5;
        float[][] vecs = TestUtils.randomlyGenerateStandardVectors(numVectors, 128, RANDOM_SEED);
        float[][] queries = TestUtils.randomlyGenerateStandardVectors(numThreads * queriesPerThread, 128, RANDOM_SEED + 99);

        IndexWriterConfig iwc = LuceneTestCase.newIndexWriterConfig();
        iwc.setUseCompoundFile(false);
        iwc.setCodec(getCodec(1, KNNConstants.DEFAULT_LEADING_SEGMENT_MERGE_DISABLED, null, new JVectorIndexQuantization.PQ(), true));
        iwc.setMergePolicy(new ForceMergesOnlyMergePolicy(false));
        iwc.setMaxBufferedDocs(-1);

        try (var fsd = FSDirectory.open(tempDir.getRoot().toPath()); var writer = new IndexWriter(fsd, iwc)) {
            for (int i = 0; i < numVectors; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(VECTOR_FIELD, vecs[i], VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc);
            }
            writer.flush();
            writer.commit();

            try (var reader = DirectoryReader.open(writer)) {
                var searcher = LuceneTestCase.newSearcher(reader);
                var latch = new CountDownLatch(1);
                var errors = new AtomicReference<Throwable>(null);
                var resultCounts = new ArrayList<Integer>(numThreads * queriesPerThread);
                for (int i = 0; i < numThreads * queriesPerThread; i++)
                    resultCounts.add(0);

                ExecutorService pool = Executors.newFixedThreadPool(numThreads);
                try {
                    for (int t = 0; t < numThreads; t++) {
                        final int threadIdx = t;
                        pool.submit(() -> {
                            try {
                                latch.await();
                                for (int q = 0; q < queriesPerThread; q++) {
                                    int idx = threadIdx * queriesPerThread + q;
                                    var query = new JVectorKnnFloatVectorQuery(VECTOR_FIELD, queries[idx], 10, 10, 0.0f, 0.0f);
                                    var results = searcher.search(query, 10).scoreDocs;
                                    resultCounts.set(idx, results.length);
                                }
                            } catch (Throwable e) {
                                errors.compareAndSet(null, e);
                            }
                        });
                    }
                    latch.countDown();
                    pool.shutdown();
                    Assert.assertTrue("Concurrent search timed out", pool.awaitTermination(60, TimeUnit.SECONDS));
                } finally {
                    pool.shutdownNow();
                }

                Assert.assertNull("Concurrent search threw: " + errors.get(), errors.get());
                for (int i = 0; i < resultCounts.size(); i++) {
                    Assert.assertEquals("Thread " + i + " got wrong result count", 10, (int) resultCounts.get(i));
                }
            }
        }
    }

    /**
     * FusedPQ merge with leading-segment merge disabled: the merge rebuilds the graph from
     * scratch but must still emit a FusedPQ-encoded segment.
     */
    @Test
    public void testFusedPQLeadingSegmentMergeDisabled() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minFusedPqThreshold(1)
            .leadingSegmentMergeDisabled(true)
            .round(MergeTestRound.builder().segmentSizes(List.of(300, 300)).build())
            .minimumRecall(0.7)
            .build();
        runScenarioWithPool(scenario, singleThreadGraphMergePool);
    }
}

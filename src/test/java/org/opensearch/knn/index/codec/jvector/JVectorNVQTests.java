/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import static org.opensearch.knn.index.engine.CommonTestUtils.getCodecWithNVQ;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.IntField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
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

import io.github.jbellis.jvector.util.DocIdSetIterator;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Builder.Default;
import lombok.Singular;

/**
 * Tests for the NVQ (Non-uniform Vector Quantization) write, merge, and search paths.
 *
 * These tests mirror the PQ scenarios in {@link JVectorWriterMergeTests} but use
 * {@link KNNConstants#QUANTIZATION_TYPE_NVQ} as the quantization type. They cover:
 * <ul>
 *   <li>Single-segment flush with NVQ enabled</li>
 *   <li>Fallback to full-precision when vector count is below the batch threshold</li>
 *   <li>Simple two-segment merge with NVQ</li>
 *   <li>Multi-segment merge with deletions under NVQ</li>
 *   <li>Multi-phase progressive merge once the vector count crosses the NVQ threshold</li>
 * </ul>
 *
 * NVQ is a lossy compressor, so recall targets are set slightly lower than the PQ
 * equivalents, and overqueryFactor is raised to compensate.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "")
public class JVectorNVQTests extends LuceneTestCase {

    static final int RANDOM_SEED = 73;
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
        int minNvqThreshold = 1; // always apply NVQ by default
        @Default
        int dimension = 128;
        @Default
        int nQueries = 10;
        @Default
        int topK = 10;
        @Default
        int overqueryFactor = 10;
        @Default
        double minimumRecall = 0.85;
        @Default
        boolean leadingSegmentMergeDisabled = KNNConstants.DEFAULT_LEADING_SEGMENT_MERGE_DISABLED;
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
        iwc.setCodec(getCodecWithNVQ(scenario.minNvqThreshold, scenario.leadingSegmentMergeDisabled, mergePool));
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
                        doc.add(new KnnFloatVectorField(VECTOR_FIELD, baseVecs[id], VectorSimilarityFunction.EUCLIDEAN));
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
                        var gt = computeL2GroundTruth(baseVecs, liveVecs, queryVec, scenario.topK);
                        var query = new JVectorKnnFloatVectorQuery(
                            VECTOR_FIELD,
                            queryVec,
                            scenario.topK,
                            scenario.overqueryFactor,
                            0.0f,
                            0.0f,
                            false
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
                        String.format("NVQ recall too low [round %d]: got %.3f < %.3f", roundId, recall, scenario.minimumRecall),
                        recall >= scenario.minimumRecall
                    );
                }
            }
        }
    }

    Set<Integer> computeL2GroundTruth(float[][] baseVectors, BitSet liveVectors, float[] query, int k) {
        if (liveVectors.length() == 0) {
            return Set.of();
        }
        var pq = new PriorityQueue<ScoreDoc>(k, (a, b) -> Float.compare(a.score, b.score));
        for (int i = liveVectors.nextSetBit(0); i != DocIdSetIterator.NO_MORE_DOCS; i = liveVectors.nextSetBit(i + 1)) {
            var score = VectorSimilarityFunction.EUCLIDEAN.compare(baseVectors[i], query);
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

    /**
     * Single segment, NVQ always active (minThreshold=1). Verifies the full
     * flush → encode → search path for NVQ.
     */
    @Test
    public void testNVQFlushRecall() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minNvqThreshold(1)
            .round(MergeTestRound.builder().segmentSizes(List.of(500)).build())
            .minimumRecall(0.85)
            .build();
        runScenario(scenario);
    }

    /**
     * NVQ type configured but minThreshold=MAX_VALUE so no quantization is ever
     * applied. Recall must be perfect since full-precision vectors are used.
     */
    @Test
    public void testNVQBelowThresholdFallbackToFullPrecision() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minNvqThreshold(Integer.MAX_VALUE)
            .round(MergeTestRound.builder().segmentSizes(List.of(200)).build())
            .minimumRecall(1.0)
            .overqueryFactor(KNNConstants.DEFAULT_OVER_QUERY_FACTOR)
            .build();
        runScenario(scenario);
    }

    /**
     * Two segments merged into one under NVQ. NVQ must recompute from scratch
     * during the merge (no refine() equivalent like PQ).
     */
    @Test
    public void testNVQSimpleMerge() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minNvqThreshold(1)
            .round(MergeTestRound.builder().segmentSizes(List.of(250, 250)).build())
            .minimumRecall(0.85)
            .build();
        runScenarioWithPool(scenario, singleThreadGraphMergePool);
    }

    /**
     * Multiple segments with deletions merged under NVQ. Tests that the recompute-from-scratch
     * merge path handles live-doc filtering correctly.
     */
    @Test
    public void testNVQMergeWithDeletions() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minNvqThreshold(1)
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(100, 200, 100, 100))
                    .deletionRanges(List.of(new DeletionRange(0, 50), new DeletionRange(300, 350)))
                    .build()
            )
            .minimumRecall(0.85)
            .build();
        runScenarioWithPool(scenario, singleThreadGraphMergePool);
    }

    /**
     * Multi-phase scenario: first two rounds stay below the NVQ threshold (full precision),
     * then the third round crosses it and NVQ is applied. Validates the progressive path
     * where earlier full-precision segments are eventually recompressed with NVQ on merge.
     */
    @Test
    public void testNVQMultiPhaseMergeProgressiveThreshold() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minNvqThreshold(600)
            .round(MergeTestRound.builder().segmentSizes(List.of(50, 100, 100)).build())   // 250 < 600, no NVQ
            .round(MergeTestRound.builder().segmentSizes(List.of(50, 100, 100)).build())   // 500 < 600, no NVQ
            .round(MergeTestRound.builder().segmentSizes(List.of(50, 100, 100)).build())   // 750 > 600, NVQ kicks in
            .overqueryFactor(20)
            .minimumRecall(0.85)
            .build();
        runScenarioWithPool(scenario, singleThreadGraphMergePool);
    }

    /**
     * Multi-phase scenario with NVQ always active and complex deletions across rounds.
     * Mirrors {@code multiPhaseMergeWithDeletesAlwaysPQ} from {@link JVectorWriterMergeTests}.
     */
    @Test
    public void testNVQMultiPhaseMergeWithDeletions() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minNvqThreshold(1)
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(100, 150, 100, 100))
                    .deletionRanges(List.of(new DeletionRange(0, 30), new DeletionRange(300, 350)))
                    .build()
            )
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(50, 100, 150))
                    .deletionRanges(List.of(new DeletionRange(400, 480), new DeletionRange(50, 55)))
                    .build()
            )
            .round(MergeTestRound.builder().segmentSizes(List.of(200)).build())
            .overqueryFactor(20)
            .minimumRecall(0.85)
            .build();
        runScenarioWithPool(scenario, singleThreadGraphMergePool);
    }

}

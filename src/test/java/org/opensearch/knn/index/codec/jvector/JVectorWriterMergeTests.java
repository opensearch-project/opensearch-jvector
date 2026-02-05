package org.opensearch.knn.index.codec.jvector;

import static org.opensearch.knn.index.engine.CommonTestUtils.getCodec;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;
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
import org.junit.Assert;
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

@ThreadLeakFilters(defaultFilters = true, filters = { ThreadLeakFiltersForTests.class })
@LuceneTestCase.SuppressSysoutChecks(bugUrl = "")
public class JVectorWriterMergeTests extends LuceneTestCase {

    static final int RANDOM_SEED = 42;
    static final String ID_FIELD = "doc_id";
    static final String VECTOR_FIELD = "vectors";
    static final String SEG_IN_ROUND = "segment_in_round";
    static final String ROUND_ID = "round_id";

    /** start inclusive, end exclusive */
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
        int minPqThreshold = KNNConstants.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION;
        @Default
        int dimension = 128;
        @Default
        int nQueries = 10;
        @Default
        int topK = 10;
        @Default
        int overqueryFactor = KNNConstants.DEFAULT_OVER_QUERY_FACTOR;
        @Default
        double minimumRecall = 0.99;
    }

    @Rule
    public TemporaryFolder tempDir = new TemporaryFolder();

    void runScenario(MergeTestScenario scenario) throws IOException {
        // don't worry about deletions when generating base vectors, they will just become holes in the ids later
        int nBase = scenario.rounds.stream().mapToInt(r -> r.segmentSizes.stream().mapToInt(x -> x).sum()).sum();
        var baseVecs = TestUtils.randomlyGenerateStandardVectors(nBase, scenario.dimension, RANDOM_SEED);
        var queryVecs = TestUtils.randomlyGenerateStandardVectors(scenario.nQueries, scenario.dimension, RANDOM_SEED + 1);

        var liveVecs = new FixedBitSet(nBase);
        liveVecs.clear();

        // Path indexPath = createTempDir();
        IndexWriterConfig iwc = LuceneTestCase.newIndexWriterConfig();
        iwc.setUseCompoundFile(false);
        iwc.setCodec(getCodec(scenario.minPqThreshold));
        iwc.setMergePolicy(new ForceMergesOnlyMergePolicy(false));

        try (var fsd = FSDirectory.open(tempDir.getRoot().toPath()); var writer = new IndexWriter(fsd, iwc);) {
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
                        doc.add(new IntField(SEG_IN_ROUND, segInRound, Store.YES));
                        doc.add(new IntField(ROUND_ID, roundId, Store.YES));
                        writer.addDocument(doc);

                        liveVecs.set(id);
                    }
                    writer.flush();
                    writer.commit();
                    vectorOffset += segmentSize;
                }

                for (var dl : round.deletionRanges) {
                    var query = IntField.newRangeQuery(ID_FIELD, dl.start, dl.end - 1); // end inclusive
                    writer.deleteDocuments(query);
                    liveVecs.clear(dl.start, dl.end);
                    writer.commit();
                }

                writer.forceMerge(1);
                writer.commit();

                try (var reader = DirectoryReader.open(writer)) {
                    // assertEquals("Should have one segment after merge", 1, reader.getContext().leaves().size());
                    Assert.assertTrue("Should have one segment after merge", 1 >= reader.getContext().leaves().size());
                    assertEquals(liveVecs.cardinality(), reader.numDocs());

                    var searcher = LuceneTestCase.newSearcher(reader);

                    var totalCorrect = 0;
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

                        var correct = 0;
                        for (var doc : pred) {
                            if (gt.contains(doc)) {
                                correct += 1;
                            }
                        }
                        totalCorrect += correct;
                    }

                    double recall = totalCorrect / (double) (scenario.topK * queryVecs.length);
                    Assert.assertTrue(
                        String.format("Recall too low [RoundId %d], got %f < %f", roundId, recall, scenario.minimumRecall),
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
            } else {
                if (pq.peek().score < sd.score) {
                    pq.poll();
                    pq.add(sd);
                }
            }
            // otherwise nextBitSet(length()) throws assertion for a FixedBitSet
            if (i + 1 >= liveVectors.length()) {
                break;
            }
        }
        return pq.stream().map(sd -> sd.doc).collect(Collectors.toSet());
    }

    @Test
    public void simpleTest() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minPqThreshold(Integer.MAX_VALUE)
            .round(MergeTestRound.builder().segmentSizes(List.of(100)).build())
            .minimumRecall(1.0)
            .build();
        runScenario(scenario);
    }

    @Test
    public void simpleMergeNoPQ() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minPqThreshold(Integer.MAX_VALUE)
            .round(MergeTestRound.builder().segmentSizes(List.of(100, 200)).build())
            .minimumRecall(1.0)
            .build();
        runScenario(scenario);
    }

    @Test
    public void manySegmentMergeNoPq() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minPqThreshold(Integer.MAX_VALUE)
            .round(MergeTestRound.builder().segmentSizes(List.of(100, 200, 50, 250, 1)).build())
            .minimumRecall(1.0)
            .build();
        runScenario(scenario);
    }

    @Test
    public void multiPhaseSegmentMergeNoPq() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minPqThreshold(Integer.MAX_VALUE)
            .round(MergeTestRound.builder().segmentSizes(List.of(10, 200, 50, 25, 1)).build())
            .round(MergeTestRound.builder().segmentSizes(List.of(10, 20, 150)).build())
            .round(
                MergeTestRound.builder()
                    // note: this segment is large, should become new leading segment
                    .segmentSizes(List.of(20, 1000, 1))
                    .build()
            )
            .overqueryFactor(20)  // can't achieve 1.0 recall otherwise
            .minimumRecall(1.0)
            .build();
        runScenario(scenario);
    }

    @Test
    public void simpleDeletionTest() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minPqThreshold(Integer.MAX_VALUE)
            .round(MergeTestRound.builder().segmentSizes(List.of(100)).deletionRanges(List.of(new DeletionRange(20, 70))).build())
            .minimumRecall(1.0)
            .build();
        runScenario(scenario);
    }

    @Test
    public void multiPhaseSegmentMergeWithDeletionsNoPq() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minPqThreshold(Integer.MAX_VALUE)
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(10, 200, 50, 100, 100))
                    .deletionRanges(
                        List.of(
                            new DeletionRange(0, 5),
                            new DeletionRange(210, 260),  // wipe out an enitre segment
                            new DeletionRange(330, 400),  // deletion range spanning segments
                            new DeletionRange(410, 411)
                        )
                    )
                    .build()
            )
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(10, 20, 300))
                    .deletionRanges(
                        List.of(
                            new DeletionRange(405, 600),  // go wild
                            new DeletionRange(10, 11)  // I just don't like index 10 anymore
                            // btw remember that deletions don't renumber ordinals as far as this test class is concerned
                            // max ordinal (and holes) just keep growing as you add more rounds
                        )
                    )
                    .build()
            )
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(20, 1000, 1))
                    .deletionRanges(List.of(new DeletionRange(998, 1023), new DeletionRange(1023, 1100), new DeletionRange(1502, 1504)))
                    .build()
            )
            .overqueryFactor(20)
            .minimumRecall(0.99)
            .build();
        runScenario(scenario);
    }

    @Test
    public void multiPhaseMergeWithDeletesProgressivePQ() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minPqThreshold(1000)   // start using PQ once the vector count crosses this
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(10, 200, 50, 100, 100))  // count 460
                    .deletionRanges(
                        List.of(
                            // total deletions 126
                            new DeletionRange(0, 5),      // count 5
                            new DeletionRange(210, 260),  // count 50
                            new DeletionRange(330, 400),  // count 70
                            new DeletionRange(410, 411)   // count 1
                        )
                    )
                    .build()  // count 334
            )
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(10, 20, 300))  // count 330
                    .deletionRanges(
                        List.of(
                            // total deletions 195
                            new DeletionRange(405, 600),  // count 195, BUT this range has some overlap so actually 194
                            new DeletionRange(10, 11)     // count 1
                        )
                    )
                    .build()  // +135, total count 469
            )
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(20, 2000, 1))  // I can't do more math, so we'll just add 2000 vectors to ensure PQ
                    .deletionRanges(List.of(new DeletionRange(998, 1023), new DeletionRange(1023, 1100), new DeletionRange(2502, 2504)))
                    .build()
            )
            .overqueryFactor(20)
            .minimumRecall(0.99)
            .build();
        runScenario(scenario);
    }

    @Test
    public void multiPhaseMergeWithDeletesAlwaysPQ() throws IOException {
        var scenario = MergeTestScenario.builder()
            .minPqThreshold(1)  // always use PQ
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(10, 200, 50, 100, 100))
                    .deletionRanges(
                        List.of(
                            new DeletionRange(0, 5),
                            new DeletionRange(210, 260),
                            new DeletionRange(330, 400),
                            new DeletionRange(410, 411)
                        )
                    )
                    .build()
            )
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(10, 20, 300))
                    .deletionRanges(List.of(new DeletionRange(405, 600), new DeletionRange(10, 11)))
                    .build()
            )
            .round(
                MergeTestRound.builder()
                    .segmentSizes(List.of(20, 2000, 1))
                    .deletionRanges(List.of(new DeletionRange(998, 1023), new DeletionRange(1023, 1100), new DeletionRange(2502, 2504)))
                    .build()
            )
            .overqueryFactor(20)
            .minimumRecall(0.99)
            .build();
        runScenario(scenario);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.FixedBitSet;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.mockito.Mockito.mock;

/**
 * Tests for KNNScorer - scorer that uses pre-computed scores from a map.
 */
public class KNNScorerTests extends KNNTestCase {

    public void testConstructorAndBasicMethods() {
        Weight mockWeight = mock(Weight.class);
        Map<Integer, Float> scores = new HashMap<>();
        scores.put(0, 0.9f);
        scores.put(1, 0.8f);
        scores.put(2, 0.7f);

        FixedBitSet bitSet = new FixedBitSet(3);
        bitSet.set(0);
        bitSet.set(1);
        bitSet.set(2);
        DocIdSetIterator iterator = new BitSetIterator(bitSet, 3);

        float boost = 1.0f;
        KNNScorer scorer = new KNNScorer(mockWeight, iterator, scores, boost);

        assertNotNull(scorer);
        assertEquals(iterator, scorer.iterator());
    }

    public void testScore_WithBoost() throws IOException {
        Weight mockWeight = mock(Weight.class);
        Map<Integer, Float> scores = new HashMap<>();
        scores.put(0, 0.5f);
        scores.put(1, 0.6f);

        FixedBitSet bitSet = new FixedBitSet(2);
        bitSet.set(0);
        bitSet.set(1);
        DocIdSetIterator iterator = new BitSetIterator(bitSet, 2);

        float boost = 2.0f;
        KNNScorer scorer = new KNNScorer(mockWeight, iterator, scores, boost);

        // Advance to first doc
        assertEquals(0, iterator.nextDoc());
        assertEquals(0, scorer.docID());
        assertEquals(0.5f * 2.0f, scorer.score(), 0.0001f);

        // Advance to second doc
        assertEquals(1, iterator.nextDoc());
        assertEquals(1, scorer.docID());
        assertEquals(0.6f * 2.0f, scorer.score(), 0.0001f);
    }

    public void testScore_WithoutBoost() throws IOException {
        Weight mockWeight = mock(Weight.class);
        Map<Integer, Float> scores = new HashMap<>();
        scores.put(0, 0.75f);

        FixedBitSet bitSet = new FixedBitSet(1);
        bitSet.set(0);
        DocIdSetIterator iterator = new BitSetIterator(bitSet, 1);

        float boost = 1.0f;
        KNNScorer scorer = new KNNScorer(mockWeight, iterator, scores, boost);

        assertEquals(0, iterator.nextDoc());
        assertEquals(0.75f, scorer.score(), 0.0001f);
    }

    public void testScore_ThrowsExceptionForMissingScore() throws IOException {
        Weight mockWeight = mock(Weight.class);
        Map<Integer, Float> scores = new HashMap<>();
        scores.put(0, 0.5f);
        // Doc 1 is missing from scores map

        FixedBitSet bitSet = new FixedBitSet(2);
        bitSet.set(0);
        bitSet.set(1);
        DocIdSetIterator iterator = new BitSetIterator(bitSet, 2);

        KNNScorer scorer = new KNNScorer(mockWeight, iterator, scores, 1.0f);

        // First doc should work
        assertEquals(0, iterator.nextDoc());
        assertEquals(0.5f, scorer.score(), 0.0001f);

        // Second doc should throw exception
        assertEquals(1, iterator.nextDoc());
        RuntimeException exception = expectThrows(RuntimeException.class, scorer::score);
        assertTrue(exception.getMessage().contains("Null score for the docID"));
    }

    public void testGetMaxScore() throws IOException {
        Weight mockWeight = mock(Weight.class);
        Map<Integer, Float> scores = new HashMap<>();
        scores.put(0, 0.5f);

        FixedBitSet bitSet = new FixedBitSet(1);
        bitSet.set(0);
        DocIdSetIterator iterator = new BitSetIterator(bitSet, 1);

        KNNScorer scorer = new KNNScorer(mockWeight, iterator, scores, 1.0f);

        assertEquals(Float.MAX_VALUE, scorer.getMaxScore(0), 0.0001f);
        assertEquals(Float.MAX_VALUE, scorer.getMaxScore(100), 0.0001f);
    }

    public void testDocID() throws IOException {
        Weight mockWeight = mock(Weight.class);
        Map<Integer, Float> scores = new HashMap<>();
        scores.put(5, 0.9f);
        scores.put(10, 0.8f);

        // Use maxDoc = 15 to ensure we have space beyond the last set bit (10)
        // This prevents FixedBitSet.nextSetBit from throwing IndexOutOfBoundsException
        FixedBitSet bitSet = new FixedBitSet(15);
        bitSet.set(5);
        bitSet.set(10);
        DocIdSetIterator iterator = new BitSetIterator(bitSet, 15);

        KNNScorer scorer = new KNNScorer(mockWeight, iterator, scores, 1.0f);

        assertEquals(-1, scorer.docID()); // Before first nextDoc()

        assertEquals(5, iterator.nextDoc());
        assertEquals(5, scorer.docID());

        assertEquals(10, iterator.nextDoc());
        assertEquals(10, scorer.docID());

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, scorer.docID());
    }

    public void testEmptyScorer() throws IOException {
        Scorer emptyScorer = KNNScorer.emptyScorer();

        assertNotNull(emptyScorer);
        DocIdSetIterator iterator = emptyScorer.iterator();
        // DocIdSetIterator.empty() returns -1 as initial docID in this Lucene version
        assertEquals(-1, iterator.docID());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
        assertEquals(0.0f, emptyScorer.getMaxScore(0), 0.0001f);
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, emptyScorer.docID());
    }

    public void testEmptyScorer_Singleton() {
        Scorer emptyScorer1 = KNNScorer.emptyScorer();
        Scorer emptyScorer2 = KNNScorer.emptyScorer();

        // Should return the same instance
        assertSame(emptyScorer1, emptyScorer2);
        assertEquals(emptyScorer1, emptyScorer2);
        assertEquals(emptyScorer1.hashCode(), emptyScorer2.hashCode());
    }

    /**
     * Helper class to create a DocIdSetIterator from a FixedBitSet
     */
    private static class BitSetIterator extends DocIdSetIterator {
        private final FixedBitSet bitSet;
        private final int maxDoc;
        private int doc = -1;

        BitSetIterator(FixedBitSet bitSet, int maxDoc) {
            this.bitSet = bitSet;
            this.maxDoc = maxDoc;
        }

        @Override
        public int docID() {
            return doc;
        }

        @Override
        public int nextDoc() {
            if (doc == NO_MORE_DOCS) {
                return NO_MORE_DOCS;
            }
            doc = bitSet.nextSetBit(doc + 1);
            if (doc >= maxDoc || doc == NO_MORE_DOCS) {
                doc = NO_MORE_DOCS;
            }
            return doc;
        }

        @Override
        public int advance(int target) {
            if (target >= maxDoc) {
                doc = NO_MORE_DOCS;
                return NO_MORE_DOCS;
            }
            doc = bitSet.nextSetBit(target);
            if (doc >= maxDoc || doc == NO_MORE_DOCS) {
                doc = NO_MORE_DOCS;
            }
            return doc;
        }

        @Override
        public long cost() {
            return bitSet.cardinality();
        }
    }
}

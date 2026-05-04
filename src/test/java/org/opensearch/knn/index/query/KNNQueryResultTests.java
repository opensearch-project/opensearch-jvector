/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.opensearch.knn.KNNTestCase;

/**
 * Tests for KNNQueryResult - a simple DTO class that holds document ID and score.
 */
public class KNNQueryResultTests extends KNNTestCase {

    public void testConstructorAndGetters() {
        int expectedId = 42;
        float expectedScore = 0.95f;

        KNNQueryResult result = new KNNQueryResult(expectedId, expectedScore);

        assertEquals(expectedId, result.getId());
        assertEquals(expectedScore, result.getScore(), 0.0001f);
    }

    public void testConstructorWithZeroValues() {
        KNNQueryResult result = new KNNQueryResult(0, 0.0f);

        assertEquals(0, result.getId());
        assertEquals(0.0f, result.getScore(), 0.0001f);
    }

    public void testConstructorWithNegativeId() {
        int negativeId = -1;
        float score = 0.5f;

        KNNQueryResult result = new KNNQueryResult(negativeId, score);

        assertEquals(negativeId, result.getId());
        assertEquals(score, result.getScore(), 0.0001f);
    }

    public void testConstructorWithNegativeScore() {
        int id = 10;
        float negativeScore = -0.5f;

        KNNQueryResult result = new KNNQueryResult(id, negativeScore);

        assertEquals(id, result.getId());
        assertEquals(negativeScore, result.getScore(), 0.0001f);
    }

    public void testConstructorWithMaxValues() {
        int maxId = Integer.MAX_VALUE;
        float maxScore = Float.MAX_VALUE;

        KNNQueryResult result = new KNNQueryResult(maxId, maxScore);

        assertEquals(maxId, result.getId());
        assertEquals(maxScore, result.getScore(), 0.0001f);
    }

    public void testConstructorWithMinValues() {
        int minId = Integer.MIN_VALUE;
        float minScore = Float.MIN_VALUE;

        KNNQueryResult result = new KNNQueryResult(minId, minScore);

        assertEquals(minId, result.getId());
        assertEquals(minScore, result.getScore(), 0.0001f);
    }
}

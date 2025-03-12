/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.codec.jvector;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene101.Lucene101Codec;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TopDocs;

import java.util.PriorityQueue;

import static org.opensearch.knn.index.codec.jvector.JVectorFormat.DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION;

public class BenchmarkCommon {
    public static final String JVECTOR_NOT_QUANTIZED = "jvector_not_quantized";
    public static final String JVECTOR_QUANTIZED = "jvector_quantized";
    public static final String LUCENE101 = "Lucene101";
    public static final String FIELD_NAME = "vector_field";

    public static Codec getCodec(String codecType) {
        return switch (codecType) {
            case JVECTOR_NOT_QUANTIZED -> new JVectorCodec(Integer.MAX_VALUE);
            case JVECTOR_QUANTIZED -> new JVectorCodec(DEFAULT_MINIMUM_BATCH_SIZE_FOR_QUANTIZATION);
            case LUCENE101 -> new Lucene101Codec();
            default -> throw new IllegalStateException("Unexpected codec type: " + codecType);
        };
    }

    // Create a wrapper class for the result
    public static class RecallResult {
        public final float recall;
        public final long timeNs;

        public RecallResult(float recall) {
            this.recall = recall;
            this.timeNs = System.nanoTime();
        }
    }

    public static float calculateRecall(TopDocs topDocs, float expectedMinScoreInTopK) {
        int relevantDocsFound = 0;
        for (int i = 0; i < topDocs.scoreDocs.length; i++) {
            if (topDocs.scoreDocs[i].score >= expectedMinScoreInTopK) {
                relevantDocsFound++;
            }
        }

        return (float) relevantDocsFound / topDocs.scoreDocs.length; // TopDocs.scoreDocs.length is K
    }

    public static float findExpectedKthMaxScore(
        float[] queryVector,
        float[][] vectors,
        VectorSimilarityFunction similarityFunction,
        int k
    ) {
        final PriorityQueue<Float> topK = new PriorityQueue<>(k);
        for (int i = 0; i < k; i++) {
            topK.add(Float.NEGATIVE_INFINITY);
        }

        for (float[] vector : vectors) {
            float score = similarityFunction.compare(queryVector, vector);
            if (score > topK.peek()) {
                topK.poll();
                topK.add(score);
            }
        }

        return topK.peek();
    }
}

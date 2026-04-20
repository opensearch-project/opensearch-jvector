/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import oshi.util.tuples.Pair;

import java.io.IOException;

public class QuantizerHelperTests extends KNNTestCase {

    public void testCalculateMeanAndStdDev() throws IOException {
        float[][] vectors = { { 1f, 2f }, { 3f, 4f }, { 5f, 6f } };
        ScalarQuantizationParams params = new ScalarQuantizationParams(ScalarQuantizationType.ONE_BIT);
        TrainingRequest<float[]> request = new MockTrainingRequest(params, vectors);
        int[] sampledIndices = { 0, 1, 2 };

        Pair<float[], float[]> result = QuantizerHelper.calculateMeanAndStdDev(request, sampledIndices);

        assertArrayEquals(new float[] { 3f, 4f }, result.getA(), 0.01f);
        assertArrayEquals(new float[] { (float) Math.sqrt(8f / 3), (float) Math.sqrt(8f / 3) }, result.getB(), 0.01f);
    }

    private static class MockTrainingRequest extends TrainingRequest<float[]> {
        private final float[][] vectors;

        public MockTrainingRequest(ScalarQuantizationParams params, float[][] vectors) {
            super(vectors.length);
            this.vectors = vectors;
        }

        @Override
        public float[] getVectorAtThePosition(int position) {
            return vectors[position];
        }
    }
}

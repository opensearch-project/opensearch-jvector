/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableMap;
import org.apache.lucene.util.Version;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.JVMLibrary;
import org.opensearch.knn.index.engine.KNNMethod;
import org.opensearch.knn.index.engine.MethodResolver;

import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * JVM Library implementation for Faiss backed by Lucene sandbox Faiss format.
 */
public class Faiss extends JVMLibrary {

    private final Map<SpaceType, Function<Float, Float>> distanceTransform;

    final static Map<String, KNNMethod> METHODS = ImmutableMap.of(METHOD_HNSW, new FaissHNSWMethod());

    private final static Map<SpaceType, Function<Float, Float>> DISTANCE_TRANSLATIONS = ImmutableMap.<
        SpaceType,
        Function<Float, Float>>builder()
        .put(SpaceType.COSINESIMIL, distance -> (2 - distance) / 2)
        .put(SpaceType.INNER_PRODUCT, distance -> distance <= 0 ? 1 / (1 - distance) : distance + 1)
        .build();

    public final static Faiss INSTANCE = new Faiss(METHODS, Version.LATEST.toString(), DISTANCE_TRANSLATIONS);

    private final MethodResolver methodResolver;

    Faiss(Map<String, KNNMethod> methods, String version, Map<SpaceType, Function<Float, Float>> distanceTransform) {
        super(methods, version);
        this.distanceTransform = distanceTransform;
        this.methodResolver = new FaissMethodResolver();
    }

    @Override
    public String getExtension() {
        throw new UnsupportedOperationException("Getting extension for Faiss is not supported");
    }

    @Override
    public String getCompoundExtension() {
        throw new UnsupportedOperationException("Getting compound extension for Faiss is not supported");
    }

    @Override
    public float score(float rawScore, SpaceType spaceType) {
        return rawScore;
    }

    @Override
    public Float distanceToRadialThreshold(Float distance, SpaceType spaceType) {
        if (this.distanceTransform.containsKey(spaceType)) {
            return this.distanceTransform.get(spaceType).apply(distance);
        }
        return spaceType.scoreTranslation(distance);
    }

    @Override
    public Float scoreToRadialThreshold(Float score, SpaceType spaceType) {
        return score;
    }

    @Override
    public org.opensearch.knn.index.engine.ResolvedMethodContext resolveMethod(
        org.opensearch.knn.index.engine.KNNMethodContext knnMethodContext,
        org.opensearch.knn.index.engine.KNNMethodConfigContext knnMethodConfigContext,
        boolean shouldRequireTraining,
        final SpaceType spaceType
    ) {
        return methodResolver.resolveMethod(knnMethodContext, knnMethodConfigContext, shouldRequireTraining, spaceType);
    }
}

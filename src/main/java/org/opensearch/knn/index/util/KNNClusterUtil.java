/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.util;

import lombok.Setter;
import lombok.NonNull;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.opensearch.Version;
import org.opensearch.action.OriginalIndices;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.search.pipeline.SearchPipelineService;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Class abstracts information related to underlying OpenSearch cluster
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
@Log4j2
public class KNNClusterUtil {

    private ClusterService clusterService;
    private IndexNameExpressionResolver indexNameExpressionResolver;
    private static KNNClusterUtil instance;
    @Setter
    private SearchPipelineService searchPipelineService;

    /**
     * Return instance of the cluster context, must be initialized first for proper usage
     * @return instance of cluster context
     */
    public static synchronized KNNClusterUtil instance() {
        if (instance == null) {
            instance = new KNNClusterUtil();
        }
        return instance;
    }

    /**
     * Initializes instance of cluster context by injecting dependencies
     * @param clusterService
     */
    public void initialize(final ClusterService clusterService, final IndexNameExpressionResolver indexNameExpressionResolver) {
        this.clusterService = clusterService;
        this.indexNameExpressionResolver = indexNameExpressionResolver;
    }

    /**
     * Return minimal OpenSearch version based on all nodes currently discoverable in the cluster
     * @return minimal installed OpenSearch version, default to Version.CURRENT which is typically the latest version
     */
    public Version getClusterMinVersion() {
        try {
            return this.clusterService.state().getNodes().getMinNodeVersion();
        } catch (Exception exception) {
            log.error(
                String.format("Failed to get cluster minimum node version, returning current node version %s instead.", Version.CURRENT),
                exception
            );
            return Version.CURRENT;
        }
    }

    /**
     * Get index metadata for the given indices
     * @param originalIndices the original indices to resolve
     * @return List of IndexMetadata for the resolved indices
     */
    public List<IndexMetadata> getIndexMetadataList(@NonNull final OriginalIndices originalIndices) {
        // Resolve the index names from the OriginalIndices
        String[] concreteIndexNames = indexNameExpressionResolver.concreteIndexNames(
            clusterService.state(),
            originalIndices.indicesOptions(),
            originalIndices.indices()
        );

        // Get metadata for each resolved index
        return Arrays.stream(concreteIndexNames)
            .map(indexName -> clusterService.state().metadata().index(indexName))
            .collect(Collectors.toList());
    }

    /**
     * Check if the system generated search processor factory is enabled or not
     * @param factoryName name of the factory
     * @return If the factory is enabled or not
     */
    public boolean isSystemGeneratedSearchFactoryEnabled(String factoryName) {
        if (searchPipelineService == null) {
            throw new IllegalStateException("search pipeline service is not initialized in the KNN cluster util.");
        }
        return searchPipelineService.isSystemGeneratedFactoryEnabled(factoryName);
    }
}

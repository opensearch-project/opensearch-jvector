/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.core.action.ActionResponse;
import org.opensearch.index.codec.CodecServiceFactory;
import org.opensearch.index.engine.EngineFactory;
import org.opensearch.knn.plugin.search.KNNConcurrentSearchRequestDecider;
import org.opensearch.knn.index.util.KNNClusterUtil;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import org.opensearch.knn.index.query.parser.KNNQueryBuilderParser;
import org.opensearch.knn.index.codec.KNNCodecService;
import org.opensearch.knn.plugin.rest.RestKNNStatsHandler;
import org.opensearch.knn.plugin.script.KNNScoringScriptEngine;
import org.opensearch.knn.plugin.stats.KNNStats;
import org.opensearch.knn.plugin.transport.KNNStatsAction;
import org.opensearch.knn.plugin.transport.KNNStatsTransportAction;
import com.google.common.collect.ImmutableList;

import org.opensearch.action.ActionRequest;
import org.opensearch.transport.client.Client;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.node.DiscoveryNodes;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.common.io.stream.NamedWriteableRegistry;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.IndexScopedSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.settings.SettingsFilter;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.env.Environment;
import org.opensearch.env.NodeEnvironment;
import org.opensearch.index.IndexModule;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.Mapper;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationStateCache;
import org.opensearch.plugins.ActionPlugin;
import org.opensearch.plugins.EnginePlugin;
import org.opensearch.plugins.ExtensiblePlugin;
import org.opensearch.plugins.MapperPlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.plugins.ScriptPlugin;
import org.opensearch.plugins.SearchPlugin;
import org.opensearch.plugins.SystemIndexPlugin;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.rest.RestController;
import org.opensearch.rest.RestHandler;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptEngine;
import org.opensearch.script.ScriptService;
import org.opensearch.search.deciders.ConcurrentSearchRequestDecider;
import org.opensearch.threadpool.ExecutorBuilder;
import org.opensearch.threadpool.FixedExecutorBuilder;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.watcher.ResourceWatcherService;

import java.security.AccessController;
import java.security.AllPermission;
import java.security.Permission;
import java.security.PrivilegedAction;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;

import static java.util.Collections.singletonList;
import static org.opensearch.knn.common.KNNConstants.KNN_THREAD_POOL_PREFIX;
import static org.opensearch.knn.common.KNNConstants.TRAIN_THREAD_POOL;

/**
 * Entry point for the KNN plugin where we define mapper for knn_vector type
 * and new query clause "knn"
 *
 *
 * Example Mapping for knn_vector type
 * "settings" : {
 *    "index": {
 *       "knn": true
 *     }
 *   },
 * "mappings": {
 *   "properties": {
 *     "my_vector": {
 *       "type": "knn_vector",
 *       "dimension": 4
 *     }
 *   }
 * }
 *
 * Example Query
 *
 *   "knn": {
 *    "my_vector": {
 *      "vector": [3, 4],
 *      "k": 3
 *    }
 *   }
 *
 */
public class JVectorKNNPlugin extends Plugin
    implements
        MapperPlugin,
        SearchPlugin,
        ActionPlugin,
        EnginePlugin,
        ScriptPlugin,
        ExtensiblePlugin,
        SystemIndexPlugin {

    public static final String LEGACY_KNN_BASE_URI = "/_opendistro/_knn";
    public static final String KNN_BASE_URI = "/_plugins/_knn";

    private ClusterService clusterService;

    public JVectorKNNPlugin() {
        super();
        /*
         * Work around to avoid security manager checks
         */
        AccessController.doPrivileged((PrivilegedAction<Object>) () -> {
            System.setSecurityManager(new SecurityManager() {

                @Override
                public void checkPermission(Permission perm) {
                    if (perm instanceof AllPermission) {
                        throw new SecurityException();
                    }
                }
            });
            return null;
        });
    }

    @Override
    public Map<String, Mapper.TypeParser> getMappers() {
        return Collections.singletonMap(KNNVectorFieldMapper.CONTENT_TYPE, new KNNVectorFieldMapper.TypeParser());
    }

    @Override
    public List<QuerySpec<?>> getQueries() {
        return singletonList(new QuerySpec<>(KNNQueryBuilder.NAME, KNNQueryBuilder::new, KNNQueryBuilderParser::fromXContent));
    }

    @Override
    public Collection<Object> createComponents(
        Client client,
        ClusterService clusterService,
        ThreadPool threadPool,
        ResourceWatcherService resourceWatcherService,
        ScriptService scriptService,
        NamedXContentRegistry xContentRegistry,
        Environment environment,
        NodeEnvironment nodeEnvironment,
        NamedWriteableRegistry namedWriteableRegistry,
        IndexNameExpressionResolver indexNameExpressionResolver,
        Supplier<RepositoriesService> repositoriesServiceSupplier
    ) {
        this.clusterService = clusterService;

        KNNSettings.state().initialize(client, clusterService);
        KNNClusterUtil.instance().initialize(clusterService);
        QuantizationStateCache.setThreadPool(threadPool);

        return ImmutableList.of(new KNNStats());
    }

    @Override
    public List<Setting<?>> getSettings() {
        return KNNSettings.state().getSettings();
    }

    public List<RestHandler> getRestHandlers(
        Settings settings,
        RestController restController,
        ClusterSettings clusterSettings,
        IndexScopedSettings indexScopedSettings,
        SettingsFilter settingsFilter,
        IndexNameExpressionResolver indexNameExpressionResolver,
        Supplier<DiscoveryNodes> nodesInCluster
    ) {

        RestKNNStatsHandler restKNNStatsHandler = new RestKNNStatsHandler();

        return ImmutableList.of(restKNNStatsHandler);
    }

    /**
     * Register action and handler so that transportClient can find proxy for action
     */
    @Override
    public List<ActionHandler<? extends ActionRequest, ? extends ActionResponse>> getActions() {
        return List.of(new ActionHandler<>(KNNStatsAction.INSTANCE, KNNStatsTransportAction.class));
    }

    @Override
    public Optional<EngineFactory> getEngineFactory(IndexSettings indexSettings) {
        return Optional.empty();
    }

    @Override
    public Optional<CodecServiceFactory> getCustomCodecServiceFactory(IndexSettings indexSettings) {
        if (indexSettings.getValue(KNNSettings.IS_KNN_INDEX_SETTING)) {
            return Optional.of(KNNCodecService::new);
        }
        return Optional.empty();
    }

    @Override
    public void onIndexModule(IndexModule indexModule) {
        KNNSettings.state().onIndexModule(indexModule);
    }

    /**
     * Sample knn custom script
     *
     * {
     *   "query": {
     *     "script_score": {
     *       "query": {
     *         "match_all": {
     *           "boost": 1
     *         }
     *       },
     *       "script": {
     *         "source": "knn_score",
     *         "lang": "knn",
     *         "params": {
     *           "field": "my_dense_vector",
     *           "vector": [
     *             1,
     *             1
     *           ]
     *         }
     *       }
     *     }
     *   }
     * }
     *
     */
    @Override
    public ScriptEngine getScriptEngine(Settings settings, Collection<ScriptContext<?>> contexts) {
        return new KNNScoringScriptEngine();
    }

    @Override
    public List<ExecutorBuilder<?>> getExecutorBuilders(Settings settings) {
        return ImmutableList.of(new FixedExecutorBuilder(settings, TRAIN_THREAD_POOL, 1, 1, KNN_THREAD_POOL_PREFIX, false));
    }

    @Override
    public Optional<ConcurrentSearchRequestDecider.Factory> getConcurrentSearchRequestDeciderFactory() {
        return Optional.of(new KNNConcurrentSearchRequestDecider.Factory());
    }
}

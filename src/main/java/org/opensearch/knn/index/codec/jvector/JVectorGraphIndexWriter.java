/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.disk.RandomAccessWriter;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.disk.*;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

import java.io.IOException;
import java.util.EnumMap;

public class JVectorGraphIndexWriter extends OnDiskGraphIndexWriter {
    private final RandomAccessWriter out;
    private final int version;
    private final long startOffset;
    private final GraphIndex graphIndex;
    private final OrdinalMapper ordinalMapper;
    private final int dimension;
    private final GraphIndex.View view;
    private final EnumMap<FeatureId, Feature> featureMap;
    private final int headerSize;

    public JVectorGraphIndexWriter(
        RandomAccessWriter out,
        int version,
        long startOffset,
        GraphIndex graph,
        OrdinalMapper oldToNewOrdinals,
        int dimension,
        EnumMap<FeatureId, Feature> features
    ) {
        super(out, version, startOffset, graph, oldToNewOrdinals, dimension, features);
        assert out != null : "RandomAccessWriter cannot be null";
        assert version >= 0 : "Version cannot be negative";
        assert startOffset >= 0 : "Start offset cannot be negative";
        assert graph != null : "Graph cannot be null";
        assert oldToNewOrdinals != null : "Ordinal mapper cannot be null";
        assert dimension > 0 : "Dimension must be greater than 0";
        assert features != null : "Features cannot be null";
        assert !features.isEmpty() : "Features cannot be empty";
        this.out = out;
        this.version = version;
        this.startOffset = startOffset;
        this.graphIndex = graph;
        this.ordinalMapper = oldToNewOrdinals;
        this.dimension = dimension;
        this.view = graph instanceof OnHeapGraphIndex ? ((OnHeapGraphIndex) graph).getFrozenView() : graph.getView();
        this.featureMap = features;
        // create a mock Header to determine the correct size
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var ch = new CommonHeader(version, dimension, 0, layerInfo, 0);
        var placeholderHeader = new Header(ch, featureMap);
        this.headerSize = placeholderHeader.size();
    }

    public static JVectorGraphIndexWriter create(
        RandomAccessWriter out,
        long startOffset,
        GraphIndex graphIndex,
        int dimension,
        EnumMap<FeatureId, Feature> features
    ) {
        var version = OnDiskGraphIndex.CURRENT_VERSION;
        var ordinalMapper = new OrdinalMapper.MapMapper(sequentialRenumbering(graphIndex));
        return new JVectorGraphIndexWriter(out, version, startOffset, graphIndex, ordinalMapper, dimension, features);
    }

    @Override
    public synchronized void writeHeader() throws IOException {
        // graph-level properties
        out.seek(startOffset);
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graphIndex, ordinalMapper);
        var commonHeader = new CommonHeader(
            version,
            dimension,
            ordinalMapper.oldToNew(view.entryNode().node),
            layerInfo,
            ordinalMapper.maxOrdinal() + 1
        );
        var header = new Header(commonHeader, featureMap);
        header.write(out);
        out.flush();
        // assert out.position() == startOffset + headerSize : String.format("%d != %d", out.position(), startOffset + headerSize);
    }
}

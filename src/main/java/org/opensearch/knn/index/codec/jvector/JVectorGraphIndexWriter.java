/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.disk.*;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.SeparatedFeature;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

import static io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter.sequentialRenumbering;

public class JVectorGraphIndexWriter implements AutoCloseable {
    private final IndexOutput outData;
    private final IndexOutput outMeta;
    private final int version;
    private final long startOffset;
    private final GraphIndex graph;
    private final OrdinalMapper ordinalMapper;
    private final int dimension;
    private final GraphIndex.View view;
    private final EnumMap<FeatureId, Feature> featureMap;
    private final int headerSize;
    private final List<Feature> inlineFeatures;

    public JVectorGraphIndexWriter(
        IndexOutput outData,
        IndexOutput outMeta,
        int version,
        long startOffset,
        GraphIndex graph,
        OrdinalMapper oldToNewOrdinals,
        int dimension,
        EnumMap<FeatureId, Feature> features
    ) {
        assert outData != null : "RandomAccessWriter cannot be null";
        assert version >= 0 : "Version cannot be negative";
        assert startOffset >= 0 : "Start offset cannot be negative";
        assert graph != null : "Graph cannot be null";
        assert oldToNewOrdinals != null : "Ordinal mapper cannot be null";
        assert dimension > 0 : "Dimension must be greater than 0";
        assert features != null : "Features cannot be null";
        assert !features.isEmpty() : "Features cannot be empty";
        this.outData = outData;
        this.outMeta = outMeta;
        this.version = version;
        this.startOffset = startOffset;
        this.graph = graph;
        this.ordinalMapper = oldToNewOrdinals;
        this.dimension = dimension;
        this.view = graph instanceof OnHeapGraphIndex ? ((OnHeapGraphIndex) graph).getFrozenView() : graph.getView();
        this.featureMap = features;
        this.inlineFeatures = features.values().stream().filter(f -> !(f instanceof SeparatedFeature)).collect(Collectors.toList());

        // create a mock Header to determine the correct size
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var ch = new CommonHeader(version, dimension, 0, layerInfo, 0);
        var placeholderHeader = new Header(ch, featureMap);
        this.headerSize = placeholderHeader.size();
    }

    public static JVectorGraphIndexWriter create(
        IndexOutput outData,
        IndexOutput outMeta,
        long startOffset,
        GraphIndex graphIndex,
        int dimension,
        EnumMap<FeatureId, Feature> features
    ) {
        var version = OnDiskGraphIndex.CURRENT_VERSION;
        var ordinalMapper = new OrdinalMapper.MapMapper(sequentialRenumbering(graphIndex));
        return new JVectorGraphIndexWriter(outData, outMeta, version, startOffset, graphIndex, ordinalMapper, dimension, features);
    }

    // @Override
    public synchronized Header getHeader() throws IOException {
        // graph-level properties
        // out.seek(startOffset);
        long currentPosition = outMeta.getFilePointer();
        var layerInfo = CommonHeader.LayerInfo.fromGraph(graph, ordinalMapper);
        var commonHeader = new CommonHeader(
            version,
            dimension,
            ordinalMapper.oldToNew(view.entryNode().node),
            layerInfo,
            ordinalMapper.maxOrdinal() + 1
        );
        return new Header(commonHeader, featureMap);
    }

    // @Override
    public synchronized Header write(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException {
        if (graph instanceof OnHeapGraphIndex) {
            var ohgi = (OnHeapGraphIndex) graph;
            if (ohgi.getDeletedNodes().cardinality() > 0) {
                throw new IllegalArgumentException("Run builder.cleanup() before writing the graph");
            }
        }
        for (var featureId : featureStateSuppliers.keySet()) {
            if (!featureMap.containsKey(featureId)) {
                throw new IllegalArgumentException(String.format("Feature %s not configured for index", featureId));
            }
        }
        if (ordinalMapper.maxOrdinal() < graph.size(0) - 1) {
            var msg = String.format(
                "Ordinal mapper from [0..%d] does not cover all nodes in the graph of size %d",
                ordinalMapper.maxOrdinal(),
                graph.size(0)
            );
            throw new IllegalStateException(msg);
        }

        // writeHeader(); // sets position to start writing features

        // for each graph node, write the associated features, followed by its neighbors at L0
        for (int newOrdinal = 0; newOrdinal <= ordinalMapper.maxOrdinal(); newOrdinal++) {
            var originalOrdinal = ordinalMapper.newToOld(newOrdinal);

            // if no node exists with the given ordinal, write a placeholder
            if (originalOrdinal == OrdinalMapper.OMITTED) {
                outData.writeInt(-1);
                for (var feature : inlineFeatures) {
                    throw new UnsupportedOperationException("Separated features are not supported in this implementation");
                    // outData.seek(outData.position() + feature.featureSize());
                }
                outData.writeInt(0);
                for (int n = 0; n < graph.getDegree(0); n++) {
                    outData.writeInt(-1);
                }
                continue;
            }

            if (!graph.containsNode(originalOrdinal)) {
                var msg = String.format("Ordinal mapper mapped new ordinal %s to non-existing node %s", newOrdinal, originalOrdinal);
                throw new IllegalStateException(msg);
            }
            outData.writeInt(newOrdinal); // unnecessary, but a reasonable sanity check
            assert outData.getFilePointer() == featureOffsetForOrdinal(newOrdinal) : String.format(
                "%d != %d",
                outData.getFilePointer(),
                featureOffsetForOrdinal(newOrdinal)
            );
            for (var feature : inlineFeatures) {
                var supplier = featureStateSuppliers.get(feature.id());
                if (supplier == null) {
                    throw new IllegalStateException("Supplier for feature " + feature.id() + " not found");
                    // outData.seek(outData.position() + feature.featureSize());
                } else {
                    feature.writeInline(new JVectorRandomAccessWriter(outData), supplier.apply(originalOrdinal));
                }
            }

            var neighbors = view.getNeighborsIterator(0, originalOrdinal);
            if (neighbors.size() > graph.getDegree(0)) {
                var msg = String.format(
                    "Node %d has more neighbors %d than the graph's max degree %d -- run Builder.cleanup()!",
                    originalOrdinal,
                    neighbors.size(),
                    graph.getDegree(0)
                );
                throw new IllegalStateException(msg);
            }
            // write neighbors list
            outData.writeInt(neighbors.size());
            int n = 0;
            for (; n < neighbors.size(); n++) {
                var newNeighborOrdinal = ordinalMapper.oldToNew(neighbors.nextInt());
                if (newNeighborOrdinal < 0 || newNeighborOrdinal > ordinalMapper.maxOrdinal()) {
                    var msg = String.format("Neighbor ordinal out of bounds: %d/%d", newNeighborOrdinal, ordinalMapper.maxOrdinal());
                    throw new IllegalStateException(msg);
                }
                outData.writeInt(newNeighborOrdinal);
            }
            assert !neighbors.hasNext();

            // pad out to maxEdgesPerNode
            for (; n < graph.getDegree(0); n++) {
                outData.writeInt(-1);
            }
        }

        // write sparse levels
        for (int level = 1; level <= graph.getMaxLevel(); level++) {
            int layerSize = graph.size(level);
            int layerDegree = graph.getDegree(level);
            int nodesWritten = 0;
            for (var it = graph.getNodes(level); it.hasNext();) {
                int originalOrdinal = it.nextInt();
                // node id
                outData.writeInt(ordinalMapper.oldToNew(originalOrdinal));
                // neighbors
                var neighbors = view.getNeighborsIterator(level, originalOrdinal);
                outData.writeInt(neighbors.size());
                int n = 0;
                for (; n < neighbors.size(); n++) {
                    outData.writeInt(ordinalMapper.oldToNew(neighbors.nextInt()));
                }
                assert !neighbors.hasNext() : "Mismatch between neighbor's reported size and actual size";
                // pad out to degree
                for (; n < layerDegree; n++) {
                    outData.writeInt(-1);
                }
                nodesWritten++;
            }
            if (nodesWritten != layerSize) {
                throw new IllegalStateException("Mismatch between layer size and nodes written");
            }
        }

        // Write separated features
        for (var featureEntry : featureMap.entrySet()) {
            if (isSeparated(featureEntry.getValue())) {
                var fid = featureEntry.getKey();
                var supplier = featureStateSuppliers.get(fid);
                if (supplier == null) {
                    throw new IllegalStateException("Supplier for feature " + fid + " not found");
                }

                // Set the offset for this feature
                var feature = (SeparatedFeature) featureEntry.getValue();
                feature.setOffset(outData.getFilePointer());

                // Write separated data for each node
                for (int newOrdinal = 0; newOrdinal <= ordinalMapper.maxOrdinal(); newOrdinal++) {
                    int originalOrdinal = ordinalMapper.newToOld(newOrdinal);
                    if (originalOrdinal != OrdinalMapper.OMITTED) {
                        feature.writeSeparately(new JVectorRandomAccessWriter(outData), supplier.apply(originalOrdinal));
                    } else {
                        throw new UnsupportedOperationException("Separated features are not supported in this implementation");
                        // outData.seek(outData.position() + feature.featureSize());
                    }
                }
            }
        }

        // Write the header again with updated offsets
        return getHeader();
    }

    private long featureOffsetForOrdinal(int ordinal) {
        int edgeSize = Integer.BYTES * (1 + graph.getDegree(0));
        long inlineBytes = ordinal * (long) (Integer.BYTES + inlineFeatures.stream().mapToInt(Feature::featureSize).sum() + edgeSize);
        return startOffset + inlineBytes // previous nodes
            + Integer.BYTES; // the ordinal of the node whose features we're about to write
    }

    /**
     * Write the index header and completed edge lists to the given output.  Inline features given in
     * `featureStateSuppliers` will also be written.  (Features that do not have a supplier are assumed
     * to have already been written by calls to writeInline).  The output IS flushed.
     * <p>
     * Each supplier takes a node ordinal and returns a FeatureState suitable for Feature.writeInline.
     */
    private boolean isSeparated(Feature feature) {
        return feature instanceof SeparatedFeature;
    }

    @Override
    public synchronized void close() throws IOException {
        view.close();
        // TODO: Those are closed from upstream, do we need to close them here in the future?
        // outMeta.close();
        // outData.close();
    }
}

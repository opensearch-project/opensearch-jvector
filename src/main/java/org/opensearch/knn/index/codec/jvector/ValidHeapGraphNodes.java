package org.opensearch.knn.index.codec.jvector;

import java.io.IOException;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import io.github.jbellis.jvector.graph.OnHeapGraphIndex;

class ValidHeapGraphNodes {
    static final int VERSION = 1;
    static final int MAGIC = 0x7A78640D;
    // TODO ensure always sorted?
    private final int[] validNodes;

    ValidHeapGraphNodes(IndexInput in) throws IOException {
        int magic = in.readInt();
        if (magic != MAGIC) {
            throw new IOException("Not the start of a ValidheapGraphNodes");
        }
        int version = in.readInt();
        if (version != VERSION) {
            throw new IOException("Unsupported version " + version);
        }
        int length = in.readInt();
        validNodes = new int[length];
        for (int i = 0; i < validNodes.length; i++) {
            validNodes[i] = in.readInt();
        }
    }

    ValidHeapGraphNodes(int[] validNodes) {
        this.validNodes = validNodes;
    }

    ValidHeapGraphNodes(OnHeapGraphIndex graph) {
        var it = graph.getNodes(0);
        validNodes = new int[it.size()];
        for (int i = 0; it.hasNext(); i++) {
            validNodes[i] = it.nextInt();
        }
    }

    int[] getValidNodes() {
        return validNodes;
    }

    void toOutput(IndexOutput out) throws IOException {
        out.writeInt(MAGIC);
        out.writeInt(VERSION);
        out.writeInt(validNodes.length);
        for (int validNode : validNodes) {
            out.writeInt(validNode);
        }
    }
}

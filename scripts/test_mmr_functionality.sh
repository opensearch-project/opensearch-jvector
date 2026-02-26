#!/bin/bash

# Test script for MMR (Maximal Marginal Relevance) functionality in jVector plugin
# Usage: ./scripts/test_mmr_functionality.sh [opensearch_url] [--verbose]
# Options:
#   --verbose, -v    Print full curl commands before execution

set -e

# Configuration
OPENSEARCH_URL="http://localhost:9200"
INDEX_NAME="test-mmr-index"
VERBOSE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        http*)
            OPENSEARCH_URL="$arg"
            shift
            ;;
    esac
done

echo "=========================================="
echo "MMR Functionality Test Script"
echo "=========================================="
echo "OpenSearch URL: $OPENSEARCH_URL"
echo "Index Name: $INDEX_NAME"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

print_section() {
    echo -e "${BLUE}━━━ $1 ━━━${NC}"
}

# Function to execute curl with optional verbose output
execute_curl() {
    local description="$1"
    shift
    
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[CURL Command]${NC}" >&2
        echo "curl $@" >&2
        echo "" >&2
    fi
    
    curl "$@"
}

# Step 1: Check OpenSearch connection
print_section "Step 1: Checking OpenSearch Connection"
if execute_curl "Check connection" -s -o /dev/null -w "%{http_code}" "$OPENSEARCH_URL" | grep -q "200\|401"; then
    print_success "OpenSearch is running"
else
    print_error "Cannot connect to OpenSearch at $OPENSEARCH_URL"
    exit 1
fi

# Step 2: Check jVector plugin
print_section "Step 2: Verifying jVector Plugin"
PLUGINS=$(execute_curl "List plugins" -s "$OPENSEARCH_URL/_cat/plugins?v")
if echo "$PLUGINS" | grep -q "jvector\|opensearch-jvector"; then
    print_success "jVector plugin is installed"
    echo "$PLUGINS"
else
    print_error "jVector plugin not found"
    exit 1
fi

# Step 3: Enable MMR processors in cluster settings
print_section "Step 3: Enabling MMR Processors"
ENABLE_RESPONSE=$(execute_curl "Enable MMR processors" -s -X PUT "$OPENSEARCH_URL/_cluster/settings?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "persistent": {
    "cluster.search.enabled_system_generated_factories": ["mmr_over_sample_factory", "mmr_rerank_factory"]
  }
}')

if echo "$ENABLE_RESPONSE" | grep -q '"acknowledged".*true'; then
    print_success "MMR processors enabled in cluster settings"
else
    print_error "Failed to enable MMR processors"
    echo "$ENABLE_RESPONSE"
    exit 1
fi

# Step 4: Clean up existing index
print_section "Step 4: Cleaning Up"
execute_curl "Delete index" -s -X DELETE "$OPENSEARCH_URL/$INDEX_NAME?pretty" > /dev/null 2>&1 || true
print_success "Cleanup complete"

# Step 5: Create index with vector field
print_section "Step 5: Creating Index with Vector Field"
CREATE_RESPONSE=$(execute_curl "Create index" -s -X PUT "$OPENSEARCH_URL/$INDEX_NAME?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "settings": {
    "index": {
      "knn": true,
      "number_of_shards": 1,
      "number_of_replicas": 0
    }
  },
  "mappings": {
    "properties": {
      "my_vector": {
        "type": "knn_vector",
        "dimension": 5,
        "method": {
          "name": "disk_ann",
          "space_type": "l2",
          "engine": "jvector",
          "parameters": {
            "ef_construction": 100,
            "m": 16
          }
        }
      },
      "title": {
        "type": "text"
      },
      "category": {
        "type": "keyword"
      }
    }
  }
}')

if echo "$CREATE_RESPONSE" | grep -q '"acknowledged".*true'; then
    print_success "Index created successfully"
else
    print_error "Failed to create index"
    echo "$CREATE_RESPONSE"
    exit 1
fi

# Step 6: Index diverse documents
print_section "Step 6: Indexing Sample Documents"

# Category A - Similar vectors (low diversity)
execute_curl "Index doc 1" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/1?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "my_vector": [1.0, 1.0, 1.0, 1.0, 1.0],
  "title": "Document A1 - Very similar to query",
  "category": "A"
}' > /dev/null

execute_curl "Index doc 2" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/2?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "my_vector": [1.1, 1.1, 1.1, 1.1, 1.1],
  "title": "Document A2 - Very similar to A1",
  "category": "A"
}' > /dev/null

execute_curl "Index doc 3" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/3?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "my_vector": [1.2, 1.2, 1.2, 1.2, 1.2],
  "title": "Document A3 - Very similar to A1 and A2",
  "category": "A"
}' > /dev/null

# Category B - Moderately different
execute_curl "Index doc 4" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/4?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "my_vector": [2.0, 2.0, 2.0, 2.0, 2.0],
  "title": "Document B1 - Moderately different",
  "category": "B"
}' > /dev/null

execute_curl "Index doc 5" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/5?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "my_vector": [2.1, 2.1, 2.1, 2.1, 2.1],
  "title": "Document B2 - Similar to B1",
  "category": "B"
}' > /dev/null

# Category C - Very different (high diversity)
execute_curl "Index doc 6" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/6?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "my_vector": [5.0, 5.0, 5.0, 5.0, 5.0],
  "title": "Document C1 - Very different",
  "category": "C"
}' > /dev/null

execute_curl "Index doc 7" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/7?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "my_vector": [0.5, 0.5, 0.5, 0.5, 0.5],
  "title": "Document D1 - Close but different direction",
  "category": "D"
}' > /dev/null

print_success "Indexed 7 documents with varying similarity"

# Step 7: Test standard KNN search (no MMR)
print_section "Step 7: Testing Standard KNN Search (No MMR)"
STANDARD_RESPONSE=$(execute_curl "Standard KNN search" -s -X GET "$OPENSEARCH_URL/$INDEX_NAME/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "size": 5,
  "query": {
    "knn": {
      "my_vector": {
        "vector": [1.0, 1.0, 1.0, 1.0, 1.0],
        "k": 5
      }
    }
  }
}')

if echo "$STANDARD_RESPONSE" | grep -q '"hits"'; then
    print_success "Standard KNN search executed"
    echo ""
    echo "Top 5 results (by relevance only):"
    echo "$STANDARD_RESPONSE" | jq -r '.hits.hits[] | "  ID: \(._id) | Score: \(._score) | Category: \(._source.category) | \(._source.title)"'
    echo ""
else
    print_error "Standard KNN search failed"
    echo "$STANDARD_RESPONSE"
fi

# Step 8: Test MMR search with low diversity (0.3)
print_section "Step 8: Testing MMR with Low Diversity (0.3)"
MMR_LOW_RESPONSE=$(execute_curl "MMR low diversity" -s -X GET "$OPENSEARCH_URL/$INDEX_NAME/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "size": 5,
  "query": {
    "knn": {
      "my_vector": {
        "vector": [1.0, 1.0, 1.0, 1.0, 1.0],
        "k": 5
      }
    }
  },
  "ext": {
    "mmr": {
      "diversity": 0.3,
      "candidates": 7
    }
  }
}')

if echo "$MMR_LOW_RESPONSE" | grep -q '"hits"'; then
    print_success "MMR search with low diversity executed"
    echo ""
    echo "Top 5 results (diversity=0.3, mostly relevance):"
    echo "$MMR_LOW_RESPONSE" | jq -r '.hits.hits[] | "  ID: \(._id) | Score: \(._score) | Category: \(._source.category) | \(._source.title)"'
    echo ""
else
    print_error "MMR search with low diversity failed"
    echo "$MMR_LOW_RESPONSE"
fi

# Step 9: Test MMR search with high diversity (0.8)
print_section "Step 9: Testing MMR with High Diversity (0.8)"
MMR_HIGH_RESPONSE=$(execute_curl "MMR high diversity" -s -X GET "$OPENSEARCH_URL/$INDEX_NAME/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "size": 5,
  "query": {
    "knn": {
      "my_vector": {
        "vector": [1.0, 1.0, 1.0, 1.0, 1.0],
        "k": 5
      }
    }
  },
  "ext": {
    "mmr": {
      "diversity": 0.8,
      "candidates": 7
    }
  }
}')

if echo "$MMR_HIGH_RESPONSE" | grep -q '"hits"'; then
    print_success "MMR search with high diversity executed"
    echo ""
    echo "Top 5 results (diversity=0.8, balanced relevance+diversity):"
    echo "$MMR_HIGH_RESPONSE" | jq -r '.hits.hits[] | "  ID: \(._id) | Score: \(._score) | Category: \(._source.category) | \(._source.title)"'
    echo ""
else
    print_error "MMR search with high diversity failed"
    echo "$MMR_HIGH_RESPONSE"
fi

# Step 10: Test MMR with explicit vector field configuration
print_section "Step 10: Testing MMR with Explicit Field Configuration"
MMR_EXPLICIT_RESPONSE=$(execute_curl "MMR explicit config" -s -X GET "$OPENSEARCH_URL/$INDEX_NAME/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "size": 3,
  "query": {
    "knn": {
      "my_vector": {
        "vector": [1.0, 1.0, 1.0, 1.0, 1.0],
        "k": 3
      }
    }
  },
  "ext": {
    "mmr": {
      "diversity": 0.5,
      "candidates": 7,
      "vector_field_path": "my_vector",
      "vector_field_data_type": "float",
      "vector_field_space_type": "l2"
    }
  }
}')

if echo "$MMR_EXPLICIT_RESPONSE" | grep -q '"hits"'; then
    print_success "MMR with explicit configuration executed"
    echo ""
    echo "Top 3 results (diversity=0.5, explicit config):"
    echo "$MMR_EXPLICIT_RESPONSE" | jq -r '.hits.hits[] | "  ID: \(._id) | Score: \(._score) | Category: \(._source.category) | \(._source.title)"'
    echo ""
else
    print_error "MMR with explicit configuration failed"
    echo "$MMR_EXPLICIT_RESPONSE"
fi

# Summary
print_section "Test Summary"
echo ""
echo "Expected Behavior:"
echo "  • Standard KNN: Returns most similar documents (A1, A2, A3, D1, B1)"
echo "    - All results from similar categories, ordered by pure relevance"
echo ""
echo "  • MMR Low Diversity (0.3): Similar to standard, slight diversity"
echo "    - Mostly relevance-focused with minor diversification"
echo ""
echo "  • MMR High Diversity (0.8): More diverse results (A1, B1, C1, D1, etc.)"
echo "    - Balanced between relevance and diversity"
echo "    - Should include documents from multiple categories (A, B, C, D)"
echo ""
print_success "All MMR tests completed!"
echo ""
print_info "Analysis:"
echo "  - Compare document IDs and categories across tests"
echo "  - Higher diversity should show more category variation"
echo "  - Scores will be recalculated by MMR algorithm"
echo ""
echo "To clean up:"
echo "  curl -X DELETE \"$OPENSEARCH_URL/$INDEX_NAME\""
echo ""

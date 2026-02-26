#!/bin/bash

# Test script for MMR (Maximal Marginal Relevance) functionality in jVector plugin
# Context: Restaurant search in San Jose with diverse cuisine types
# Usage: ./scripts/test_mmr_restaurants.sh [opensearch_url] [--verbose]
# Options:
#   --verbose, -v    Print full curl commands before execution

set -e

# Configuration
OPENSEARCH_URL="http://localhost:9200"
INDEX_NAME="san-jose-restaurants"
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
echo "MMR Restaurant Search Test - San Jose"
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

# Step 5: Create index with vector field for restaurant embeddings
print_section "Step 5: Creating Restaurant Index"
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
      "restaurant_embedding": {
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
      "name": {
        "type": "text"
      },
      "cuisine": {
        "type": "keyword"
      },
      "rating": {
        "type": "float"
      },
      "price_range": {
        "type": "keyword"
      },
      "location": {
        "type": "keyword"
      }
    }
  }
}')

if echo "$CREATE_RESPONSE" | grep -q '"acknowledged".*true'; then
    print_success "Restaurant index created successfully"
else
    print_error "Failed to create index"
    echo "$CREATE_RESPONSE"
    exit 1
fi

# Step 6: Index diverse restaurants in San Jose
print_section "Step 6: Indexing San Jose Restaurants"

# Italian restaurants (similar embeddings - cluster 1)
execute_curl "Index restaurant 1" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/1?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "restaurant_embedding": [1.0, 1.0, 1.0, 1.0, 1.0],
  "name": "Paesano Ristorante Italiano",
  "cuisine": "Italian",
  "rating": 4.5,
  "price_range": "$$$",
  "location": "Downtown San Jose"
}' > /dev/null

execute_curl "Index restaurant 2" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/2?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "restaurant_embedding": [1.1, 1.1, 1.1, 1.1, 1.1],
  "name": "Maggianos Little Italy",
  "cuisine": "Italian",
  "rating": 4.3,
  "price_range": "$$",
  "location": "Santana Row"
}' > /dev/null

execute_curl "Index restaurant 3" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/3?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "restaurant_embedding": [1.2, 1.2, 1.2, 1.2, 1.2],
  "name": "Osteria Coppa",
  "cuisine": "Italian",
  "rating": 4.4,
  "price_range": "$$$",
  "location": "San Pedro Square"
}' > /dev/null

# Japanese restaurants (moderately different - cluster 2)
execute_curl "Index restaurant 4" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/4?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "restaurant_embedding": [2.0, 2.0, 2.0, 2.0, 2.0],
  "name": "Sushi Confidential",
  "cuisine": "Japanese",
  "rating": 4.6,
  "price_range": "$$$",
  "location": "Japantown"
}' > /dev/null

execute_curl "Index restaurant 5" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/5?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "restaurant_embedding": [2.1, 2.1, 2.1, 2.1, 2.1],
  "name": "Minato Japanese Restaurant",
  "cuisine": "Japanese",
  "rating": 4.4,
  "price_range": "$$",
  "location": "Japantown"
}' > /dev/null

# Mexican restaurants (very different - cluster 3)
execute_curl "Index restaurant 6" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/6?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "restaurant_embedding": [5.0, 5.0, 5.0, 5.0, 5.0],
  "name": "La Victoria Taqueria",
  "cuisine": "Mexican",
  "rating": 4.2,
  "price_range": "$",
  "location": "Downtown San Jose"
}' > /dev/null

# Vietnamese restaurants (different direction - cluster 4)
execute_curl "Index restaurant 7" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/7?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "restaurant_embedding": [0.5, 0.5, 0.5, 0.5, 0.5],
  "name": "Pho Ha Noi",
  "cuisine": "Vietnamese",
  "rating": 4.5,
  "price_range": "$",
  "location": "East San Jose"
}' > /dev/null

# Indian restaurants (another diverse option - cluster 5)
execute_curl "Index restaurant 8" -s -X POST "$OPENSEARCH_URL/$INDEX_NAME/_doc/8?refresh=true" \
  -H 'Content-Type: application/json' \
  -d '{
  "restaurant_embedding": [3.5, 3.5, 3.5, 3.5, 3.5],
  "name": "Amber India",
  "cuisine": "Indian",
  "rating": 4.3,
  "price_range": "$$",
  "location": "Santana Row"
}' > /dev/null

print_success "Indexed 8 restaurants with diverse cuisines"

# Step 7: Test standard KNN search (no MMR)
print_section "Step 7: Standard Search - Looking for Italian-style restaurants"
print_info "Query: User wants Italian food (embedding similar to Italian restaurants)"
STANDARD_RESPONSE=$(execute_curl "Standard KNN search" -s -X GET "$OPENSEARCH_URL/$INDEX_NAME/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "size": 5,
  "query": {
    "knn": {
      "restaurant_embedding": {
        "vector": [1.0, 1.0, 1.0, 1.0, 1.0],
        "k": 5
      }
    }
  }
}')

if echo "$STANDARD_RESPONSE" | grep -q '"hits"'; then
    print_success "Standard KNN search executed"
    echo ""
    echo "Top 5 results (by relevance only - all Italian!):"
    echo "$STANDARD_RESPONSE" | jq -r '.hits.hits[] | "  \(._source.name) | Cuisine: \(._source.cuisine) | Rating: \(._source.rating) | Price: \(._source.price_range) | Score: \(._score)"'
    echo ""
    print_info "Problem: All results are Italian - no variety!"
else
    print_error "Standard KNN search failed"
    echo "$STANDARD_RESPONSE"
fi

# Step 8: Test MMR search with low diversity (0.3)
print_section "Step 8: MMR Search with Low Diversity (0.3)"
print_info "Still mostly Italian, but starting to see some variety"
MMR_LOW_RESPONSE=$(execute_curl "MMR low diversity" -s -X GET "$OPENSEARCH_URL/$INDEX_NAME/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "size": 5,
  "query": {
    "knn": {
      "restaurant_embedding": {
        "vector": [1.0, 1.0, 1.0, 1.0, 1.0],
        "k": 5
      }
    }
  },
  "ext": {
    "mmr": {
      "diversity": 0.3,
      "candidates": 8
    }
  }
}')

if echo "$MMR_LOW_RESPONSE" | grep -q '"hits"'; then
    print_success "MMR search with low diversity executed"
    echo ""
    echo "Top 5 results (diversity=0.3, mostly relevance):"
    echo "$MMR_LOW_RESPONSE" | jq -r '.hits.hits[] | "  \(._source.name) | Cuisine: \(._source.cuisine) | Rating: \(._source.rating) | Price: \(._source.price_range) | Score: \(._score)"'
    echo ""
else
    print_error "MMR search with low diversity failed"
    echo "$MMR_LOW_RESPONSE"
fi

# Step 9: Test MMR search with high diversity (0.8)
print_section "Step 9: MMR Search with High Diversity (0.8)"
print_info "Balanced results - Italian + diverse cuisines!"
MMR_HIGH_RESPONSE=$(execute_curl "MMR high diversity" -s -X GET "$OPENSEARCH_URL/$INDEX_NAME/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "size": 5,
  "query": {
    "knn": {
      "restaurant_embedding": {
        "vector": [1.0, 1.0, 1.0, 1.0, 1.0],
        "k": 5
      }
    }
  },
  "ext": {
    "mmr": {
      "diversity": 0.8,
      "candidates": 8
    }
  }
}')

if echo "$MMR_HIGH_RESPONSE" | grep -q '"hits"'; then
    print_success "MMR search with high diversity executed"
    echo ""
    echo "Top 5 results (diversity=0.8, balanced relevance + diversity):"
    echo "$MMR_HIGH_RESPONSE" | jq -r '.hits.hits[] | "  \(._source.name) | Cuisine: \(._source.cuisine) | Rating: \(._source.rating) | Price: \(._source.price_range) | Score: \(._score)"'
    echo ""
    print_info "Notice: Mix of Italian, Japanese, Mexican, Vietnamese, Indian!"
else
    print_error "MMR search with high diversity failed"
    echo "$MMR_HIGH_RESPONSE"
fi

# Step 10: Test MMR with explicit configuration
print_section "Step 10: MMR with Explicit Field Configuration"
print_info "Top 5 diverse restaurants with explicit vector field config"
MMR_EXPLICIT_RESPONSE=$(execute_curl "MMR explicit config" -s -X GET "$OPENSEARCH_URL/$INDEX_NAME/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
  "size": 5,
  "query": {
    "knn": {
      "restaurant_embedding": {
        "vector": [1.0, 1.0, 1.0, 1.0, 1.0],
        "k": 5
      }
    }
  },
  "ext": {
    "mmr": {
      "diversity": 0.5,
      "candidates": 8,
      "vector_field_path": "restaurant_embedding",
      "vector_field_data_type": "float",
      "vector_field_space_type": "l2"
    }
  }
}')

if echo "$MMR_EXPLICIT_RESPONSE" | grep -q '"hits"'; then
    print_success "MMR with explicit configuration executed"
    echo ""
    echo "Top 5 results (diversity=0.5, explicit config):"
    echo "$MMR_EXPLICIT_RESPONSE" | jq -r '.hits.hits[] | "  \(._source.name) | Cuisine: \(._source.cuisine) | Rating: \(._source.rating) | Price: \(._source.price_range) | Score: \(._score)"'
    echo ""
else
    print_error "MMR with explicit configuration failed"
    echo "$MMR_EXPLICIT_RESPONSE"
fi

# Summary
print_section "Test Summary - Restaurant Search Scenario"
echo ""
echo "🍝 Scenario: User searches for Italian-style restaurants in San Jose"
echo ""
echo "📊 Results Comparison:"
echo ""
echo "  1️⃣  Standard KNN (No MMR):"
echo "      → Returns: Paesano, Maggianos, Osteria Coppa (all Italian)"
echo "      → Problem: No variety - user might want to explore other options"
echo ""
echo "  2️⃣  MMR Low Diversity (0.3):"
echo "      → Returns: Mostly Italian with slight variation"
echo "      → Use case: User strongly prefers Italian but open to similar"
echo ""
echo "  3️⃣  MMR High Diversity (0.8):"
echo "      → Returns: Italian + Japanese + Mexican + Vietnamese + Indian"
echo "      → Use case: User wants variety - explore different cuisines"
echo "      → Benefit: Discovers great restaurants they might have missed!"
echo ""
echo "💡 Real-world Benefits:"
echo "   • Prevents 'filter bubble' - shows diverse options"
echo "   • Better user experience - variety in recommendations"
echo "   • Increases discovery - users find new favorites"
echo "   • Balances relevance with exploration"
echo ""
print_success "All MMR restaurant search tests completed!"
echo ""
print_info "Key Insight:"
echo "  MMR helps users discover diverse restaurants while still prioritizing"
echo "  their preferences. Perfect for recommendation systems!"
echo ""
echo "To clean up:"
echo "  curl -X DELETE \"$OPENSEARCH_URL/$INDEX_NAME\""
echo ""

# Made with Bob

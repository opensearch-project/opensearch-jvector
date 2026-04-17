#!/usr/bin/env bash
# =============================================================================
# OpenSearch jVector Plugin — Demo Script
#
# Walks through the core jVector workflow:
#   1. Create an index with a knn_vector mapping
#   2. Bulk-index documents that have vector + metadata fields
#   3. Basic KNN search
#   4. Filtered KNN search (combine vector similarity with a term filter)
#   5. KNN search with tuning parameters (overquery_factor, rerank_floor, …)
#   6. Force merge  ← triggers jVector's incremental graph merge
#   7. KNN search after merge  ← verify results are consistent
#   8. jVector node stats
#
# Usage:
#   ./scripts/demo.sh [OPTIONS]
#
# Options:
#   -h HOST   OpenSearch host:port   (default: localhost:9200)
#   -u USER   Basic-auth username    (optional)
#   -p PASS   Basic-auth password    (optional)
#   -s        Use HTTPS              (default: HTTP)
#   -x        Keep the demo index after the run (default: delete it)
# =============================================================================

set -euo pipefail

# ---------- defaults ----------------------------------------------------------
HOST="localhost:9200"
SCHEME="http"
AUTH_USER=""
AUTH_PASS=""
SKIP_CLEANUP=false
INDEX="jvector-demo"

# ---------- argument parsing --------------------------------------------------
while getopts "h:u:p:sx" opt; do
  case $opt in
    h) HOST="$OPTARG"      ;;
    u) AUTH_USER="$OPTARG" ;;
    p) AUTH_PASS="$OPTARG" ;;
    s) SCHEME="https"      ;;
    x) SKIP_CLEANUP=true   ;;
    *) echo "Unknown option -$OPTARG" >&2; exit 1 ;;
  esac
done

BASE_URL="${SCHEME}://${HOST}"
CURL_FLAGS="-sk"
[[ -n "$AUTH_USER" && -n "$AUTH_PASS" ]] && CURL_FLAGS="$CURL_FLAGS -u ${AUTH_USER}:${AUTH_PASS}"
CURL="curl ${CURL_FLAGS}"

# ---------- pretty-print helper -----------------------------------------------
pp() {
  if command -v python3 &>/dev/null; then
    python3 -m json.tool 2>/dev/null || cat
  else
    cat
  fi
}

# ---------- section header helper ---------------------------------------------
section() { printf "\n\033[1;34m=== %s ===\033[0m\n" "$*"; }
ok()      { printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }
info()    { printf "  %s\n" "$*"; }

# =============================================================================
section "0 · Cluster health check"
# =============================================================================

health=$($CURL "${BASE_URL}/_cluster/health")
status=$(echo "$health" | grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4)
version=$($CURL "${BASE_URL}" | grep -o '"number" *: *"[^"]*"' | head -1 | grep -o '[0-9][^"]*' || true)
info "OpenSearch version : ${version:-unknown}"
info "Cluster status     : ${status:-unknown}"

plugins=$($CURL "${BASE_URL}/_cat/plugins?h=component" 2>/dev/null || true)
if echo "$plugins" | grep -qi "jvector" 2>/dev/null; then
  ok "jVector plugin detected"
else
  printf "\033[1;31m  ✗ jVector plugin not found — install it before running this demo\033[0m\n"
  printf "  See DEVELOPER_GUIDE.md for installation instructions.\n"
  exit 1
fi

if [[ "${status:-}" == "red" ]]; then
  echo "Cluster is RED — aborting." >&2; exit 1
fi

# =============================================================================
section "1 · Delete previous demo index (if any)"
# =============================================================================

$CURL -X DELETE "${BASE_URL}/${INDEX}" >/dev/null 2>&1 || true
ok "Old index removed (or did not exist)"

# =============================================================================
section "2 · Create index"
# =============================================================================
#
# This demo stores 8-dimensional embeddings for a small movie catalogue.
# The 8 dimensions loosely represent:
#   [action, drama, comedy, sci-fi, romance, thriller, horror, animation]
#
# We use cosinesimil so the ranking is based on angle, not magnitude.
# =============================================================================

$CURL -X PUT "${BASE_URL}/${INDEX}" \
  -H "Content-Type: application/json" -d '
{
  "settings": {
    "index.knn": true,
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "title":    { "type": "text"    },
      "genre":    { "type": "keyword" },
      "year":     { "type": "integer" },
      "embedding": {
        "type": "knn_vector",
        "dimension": 8,
        "method": {
          "name":       "disk_ann",
          "engine":     "jvector",
          "space_type": "cosinesimil",
          "parameters": {
            "m":               16,
            "ef_construction": 100
          }
        }
      }
    }
  }
}' | pp
ok "Index '${INDEX}' created"

# =============================================================================
section "3 · Bulk-index 10 movie documents"
# =============================================================================

$CURL -X POST "${BASE_URL}/_bulk" \
  -H "Content-Type: application/json" -d '
{ "index": { "_index": "jvector-demo", "_id": "1" } }
{ "title": "The Dark Knight",          "genre": "action",    "year": 2008, "embedding": [0.9, 0.6, 0.1, 0.2, 0.0, 0.7, 0.1, 0.0] }
{ "index": { "_index": "jvector-demo", "_id": "2" } }
{ "title": "Inception",                "genre": "scifi",     "year": 2010, "embedding": [0.7, 0.5, 0.1, 0.9, 0.2, 0.6, 0.0, 0.0] }
{ "index": { "_index": "jvector-demo", "_id": "3" } }
{ "title": "The Shawshank Redemption", "genre": "drama",     "year": 1994, "embedding": [0.1, 0.9, 0.3, 0.0, 0.4, 0.2, 0.0, 0.0] }
{ "index": { "_index": "jvector-demo", "_id": "4" } }
{ "title": "Toy Story",                "genre": "animation", "year": 1995, "embedding": [0.2, 0.4, 0.8, 0.1, 0.3, 0.0, 0.0, 0.9] }
{ "index": { "_index": "jvector-demo", "_id": "5" } }
{ "title": "Interstellar",             "genre": "scifi",     "year": 2014, "embedding": [0.5, 0.6, 0.0, 0.9, 0.1, 0.4, 0.0, 0.0] }
{ "index": { "_index": "jvector-demo", "_id": "6" } }
{ "title": "The Notebook",             "genre": "romance",   "year": 2004, "embedding": [0.0, 0.7, 0.4, 0.0, 0.9, 0.1, 0.0, 0.0] }
{ "index": { "_index": "jvector-demo", "_id": "7" } }
{ "title": "Get Out",                  "genre": "horror",    "year": 2017, "embedding": [0.3, 0.5, 0.0, 0.2, 0.0, 0.8, 0.9, 0.0] }
{ "index": { "_index": "jvector-demo", "_id": "8" } }
{ "title": "Mad Max: Fury Road",       "genre": "action",    "year": 2015, "embedding": [0.9, 0.3, 0.0, 0.3, 0.0, 0.5, 0.2, 0.0] }
{ "index": { "_index": "jvector-demo", "_id": "9" } }
{ "title": "Finding Nemo",             "genre": "animation", "year": 2003, "embedding": [0.1, 0.5, 0.7, 0.0, 0.5, 0.0, 0.0, 0.9] }
{ "index": { "_index": "jvector-demo", "_id": "10" } }
{ "title": "A Beautiful Mind",         "genre": "drama",     "year": 2001, "embedding": [0.1, 0.8, 0.2, 0.3, 0.3, 0.3, 0.0, 0.0] }
' | pp

$CURL -X POST "${BASE_URL}/${INDEX}/_refresh" >/dev/null
ok "10 documents indexed and refreshed"

# =============================================================================
section "4 · Basic KNN search"
# =============================================================================
#
# Query embedding is biased toward sci-fi + action.
# Expected top-3: Inception, Interstellar, The Dark Knight.
# =============================================================================

info "Query vector → sci-fi/action bias: [0.8, 0.4, 0.0, 0.9, 0.0, 0.3, 0.0, 0.0]"

$CURL -X POST "${BASE_URL}/${INDEX}/_search" \
  -H "Content-Type: application/json" -d '
{
  "size": 3,
  "_source": ["title", "genre", "year"],
  "query": {
    "knn": {
      "embedding": {
        "vector": [0.8, 0.4, 0.0, 0.9, 0.0, 0.3, 0.0, 0.0],
        "k": 3
      }
    }
  }
}' | pp
ok "Basic KNN search done"

# =============================================================================
section "5 · Filtered KNN search (action genre only)"
# =============================================================================

info "Same query vector, restricted to genre=action"

$CURL -X POST "${BASE_URL}/${INDEX}/_search" \
  -H "Content-Type: application/json" -d '
{
  "size": 3,
  "_source": ["title", "genre", "year"],
  "query": {
    "knn": {
      "embedding": {
        "vector": [0.8, 0.4, 0.0, 0.9, 0.0, 0.3, 0.0, 0.0],
        "k": 3,
        "filter": {
          "term": { "genre": "action" }
        }
      }
    }
  }
}' | pp
ok "Filtered search done"

# =============================================================================
section "6 · KNN search with tuning parameters"
# =============================================================================
#
# overquery_factor=10 means jVector fetches 10× the requested k as internal
# candidates before re-ranking, trading latency for recall.
# =============================================================================

info "Animation-biased query with overquery_factor=10 and rerank_floor=0.1"

$CURL -X POST "${BASE_URL}/${INDEX}/_search" \
  -H "Content-Type: application/json" -d '
{
  "size": 3,
  "_source": ["title", "genre", "year"],
  "query": {
    "knn": {
      "embedding": {
        "vector": [0.1, 0.3, 0.7, 0.0, 0.4, 0.0, 0.0, 0.9],
        "k": 3,
        "method_parameters": {
          "overquery_factor": 10,
          "advanced.rerank_floor": 0.1
        }
      }
    }
  }
}' | pp
ok "Tuned search done"

# =============================================================================
section "7 · Force merge"
# =============================================================================
#
# Consolidates segments and triggers jVector's incremental graph merge.
# On a small index the effect is minimal; on millions of docs this is
# the step that unlocks full DiskANN performance.
# =============================================================================

info "Running _forcemerge?max_num_segments=1 …"
$CURL -X POST "${BASE_URL}/${INDEX}/_forcemerge?max_num_segments=1&wait_for_completion=true" | pp
$CURL -X POST "${BASE_URL}/${INDEX}/_refresh" >/dev/null
ok "Force merge complete"

# =============================================================================
section "8 · Repeat sci-fi/action search after merge"
# =============================================================================

info "Same query as step 4 — results should be consistent after merge"

$CURL -X POST "${BASE_URL}/${INDEX}/_search" \
  -H "Content-Type: application/json" -d '
{
  "size": 3,
  "_source": ["title", "genre", "year"],
  "query": {
    "knn": {
      "embedding": {
        "vector": [0.8, 0.4, 0.0, 0.9, 0.0, 0.3, 0.0, 0.0],
        "k": 3
      }
    }
  }
}' | pp
ok "Post-merge search done"

# =============================================================================
section "9 · jVector KNN node stats"
# =============================================================================
#
# knn_query_visited_nodes       — total graph nodes visited across all searches
# knn_query_expanded_nodes      — nodes whose edges were explored
# knn_query_expanded_base_layer_nodes — base-layer expansions
# =============================================================================

$CURL "${BASE_URL}/_plugins/_knn/stats?stat=knn_query_visited_nodes,knn_query_expanded_nodes,knn_query_expanded_base_layer_nodes,knn_graph_merge_time,knn_quantization_training_time" | pp
ok "Stats fetched"

# =============================================================================
section "10 · Cleanup"
# =============================================================================

if [[ "$SKIP_CLEANUP" == "true" ]]; then
  info "Index '${INDEX}' left in place (-x flag set)"
  info "Explore it: curl '${BASE_URL}/${INDEX}/_search?pretty'"
else
  $CURL -X DELETE "${BASE_URL}/${INDEX}" | pp
  ok "Demo index deleted"
fi

printf "\n\033[1;32mDemo complete!\033[0m\n\n"
printf "Next steps:\n"
printf "  • Full parameter reference : docs/user_guide.md\n"
printf "  • MMR / diverse search     : docs/mmr_search.md\n"
printf "  • Large-scale benchmarking : scripts/jvector_index_and_search/\n\n"

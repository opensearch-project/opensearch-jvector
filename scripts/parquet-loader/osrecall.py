#!/usr/bin/env python3
import argparse
import json
import os
import socket
import statistics
import sys
import time
from typing import Dict, List, Set

from dotenv import load_dotenv
import urllib3

urllib3.disable_warnings()

import numpy as np
from opensearchpy import OpenSearch, Urllib3HttpConnection
from tqdm import tqdm

# ------------- recall math -------------
def calculate_recall(truth: Set[str], recalled: Set[str]) -> float:
    if not truth:
        return 0.0
    missed = truth - recalled
    return (1.0 - (float(len(missed)) / float(len(truth)))) * 100.0


# ------------- OpenSearch client -------------
def get_opensearch_client(max_retries: int = 5, pool_maxsize: int = 4):
    """
    Configure via env or args:
      OPENSEARCH_HOSTS='[{"host":"localhost","port":9200,"scheme":"http"}]'
      OPENSEARCH_USER=...  OPENSEARCH_PASS=...
      OPENSEARCH_SSL_VERIFY=true|false
    """
    hosts = json.loads(os.getenv("OPENSEARCH_HOSTS", '[{"host":"localhost","port":9200,"scheme":"http"}]'))
    user = os.getenv("OPENSEARCH_USER")
    pwd = os.getenv("OPENSEARCH_PASS")
    http_auth = (user, pwd) if user and pwd else None
    verify_certs = os.getenv("OPENSEARCH_SSL_VERIFY", "false").lower() == "true"

    return OpenSearch(
        hosts=hosts,
        http_auth=http_auth,
        verify_certs=verify_certs,
        http_compress=True,
        retry_on_timeout=True,
        max_retries=max_retries,
        timeout=30,
        connection_class=Urllib3HttpConnection,
        pool_maxsize=pool_maxsize,
    )


def connection_check(client: OpenSearch, debug: bool = False):
    # DNS preflight to catch typos fast
    try:
        hosts = json.loads(os.getenv("OPENSEARCH_HOSTS", "[]"))
        for h in hosts:
            host = h.get("host")
            if host:
                socket.getaddrinfo(host, None)
    except Exception as e:
        print(f"❌ DNS resolution failed: {e}", file=sys.stderr)
        if debug:
            import traceback; traceback.print_exc()
        sys.exit(2)
    # Ping + info
    try:
        if not client.ping(params={"request_timeout": 5}):
            raise RuntimeError("Ping returned False (URL/creds/cluster?)")
        _ = client.info()
    except Exception as e:
        print(f"❌ Connection check failed: {e}", file=sys.stderr)
        if debug:
            import traceback; traceback.print_exc()
        sys.exit(2)


# ------------- OpenSearch queries -------------
def do_search(client: OpenSearch, index: str, field: str, query_vec: List[float], k: int, overquery_factor: int):
    """
    POST /{index}/_search with a knn query
    """
    body = {
        "size": k,
        "query": {
            "knn": {
                field: {
                    "vector": query_vec,
                    "k": k * overquery_factor,
#                    "method_parameters": {
#                        "overquery_factor": overquery_factor,
#                        "advanced.threshold": 0.0,
#                        "advanced.rerank_floor": 0.0
#                    }
                }
            }
        }
    }
    resp = client.search(index=index, body=body)
    hits = resp.get("hits", {}).get("hits", [])
    return hits




# ------------- Runner -------------
def run(args):
    # Load truth
    with open(args.truth_file) as fd:
        truth_json = json.load(fd)

    query_points: Dict[int, List[float]] = {}
    nn_ids: Dict[int, List[str]] = {}
    for i, qp in enumerate(truth_json.get("query_points", [])):
        query_points[i] = qp["point"]
        nn_ids[i] = qp["nn_id"]

    if not query_points:
        print("No query_points found in truth file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(query_points)} queries; first has {len(nn_ids[0])} truth neighbors")


    # OS client
    client = get_opensearch_client(max_retries=args.max_retries, pool_maxsize=args.pool_maxsize)
    connection_check(client, debug=args.debug)


    # Warm-up
    print("Warming up...")
    warm_keys = list(query_points.keys())[: min(5, len(query_points))]
    for qpidx in warm_keys:
        _ = do_search(client, args.index, args.field, query_points[qpidx], args.k, args.refine)

    # Measure
    print("Running measured queries...")
    recalls: List[float] = []
    latencies: List[float] = []

    iterator = query_points.keys()
    if args.limit and args.limit > 0:
        iterator = list(iterator)[: args.limit]

    for qpidx in tqdm(iterator, total=(args.limit or len(query_points)), desc="Queries"):
        qp = query_points[qpidx]
        t0 = time.perf_counter()
        hits = do_search(client, args.index, args.field, qp, args.k, args.refine)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

        # gather returned ids
        got_ids = {h.get("_id") for h in hits if h.get("_id") is not None}

        # truth ids for this query
        truth_ids = set(map(str, nn_ids[qpidx]))  # ensure comparable types

        r = calculate_recall(truth_ids, got_ids)
        recalls.append(r)

        if args.verbose:
            print(f"q={qpidx} recall={r:.2f}  hits={len(got_ids)}  latency={(t1-t0)*1000:.1f}ms")

    # Summary
    if recalls:
        print(f"\nmean recall:   {np.mean(recalls):.2f} %")
        print(f"p50 recall:    {np.percentile(recalls, 50):.2f} %")
        print(f"p95 recall:    {np.percentile(recalls, 95):.2f} %")
    if latencies:
        ms = [x * 1000.0 for x in latencies]
        print(f"mean latency:  {statistics.mean(ms):.1f} ms")
        print(f"p50 latency:   {np.percentile(ms, 50):.1f} ms")
        print(f"p95 latency:   {np.percentile(ms, 95):.1f} ms")
        print(f"qps (approx):  {len(latencies) / sum(latencies):.1f} q/s")


# ------------- CLI -------------
if __name__ == "__main__":
    load_dotenv()

    ap = argparse.ArgumentParser("Recall tester for OpenSearch k-NN")
    ap.add_argument("--truth_file", type=str, required=True, help="Ground-truth JSON")
    ap.add_argument("--index", type=str, default=os.getenv("OPENSEARCH_INDEX"), help="OpenSearch index name")
    ap.add_argument("--field", type=str, default="embeddings", help="Vector field name")
    ap.add_argument("-k", type=int, default=100, help="Top-K to return")
    ap.add_argument("--num_candidates", type=int, default=None, help="Approx. candidates (>= k). If unset, uses k*refine")
    ap.add_argument("--refine", type=int, default=20, help="Fallback multiplier for num_candidates (k*refine)")
    ap.add_argument("--api", choices=["search", "knn_search"], default="search", help="Which API to use")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of queries (0 = all)")
    ap.add_argument("--max_retries", type=int, default=5, help="Client transport retries")
    ap.add_argument("--pool_maxsize", type=int, default=4, help="HTTP connection pool per process")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not args.index:
        print("❌ Missing --index or OPENSEARCH_INDEX", file=sys.stderr)
        sys.exit(1)

    run(args)

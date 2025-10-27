#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import socket
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
from threading import Thread
from queue import Empty
import multiprocessing as mp

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm

import urllib3

urllib3.disable_warnings()

from opensearchpy import OpenSearch, Urllib3HttpConnection, TransportError
from opensearchpy.helpers import streaming_bulk

from dotenv import load_dotenv

# ------------------------------
# Optional ultra-fast JSON (NumPy → JSON without .tolist())
# ------------------------------
try:
    import orjson
    from opensearchpy.serializer import JSONSerializer

    class ORJSONSerializer(JSONSerializer):
        def dumps(self, data):
            if isinstance(data, (bytes, str)):
                return data
            return orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
except Exception:
    ORJSONSerializer = None

PROG_ADD = "add"         # increment docs by n
PROG_FILE = "file_done"  # one file finished
PROG_DONE = "done"       # one worker finished

def start_progress_consumer(progress_q, total_docs: int, total_files: int, procs: int):
    """
    Run a background thread that updates two bars:
      - docs indexed (primary, by actions confirmed from workers)
      - files done (secondary, optional)
    """
    def _runner():
        doc_bar = tqdm(total=total_docs or 0, desc="Indexed docs", unit="doc", position=0)
        file_bar = tqdm(total=total_files or 0, desc="Files", unit="file", position=1, leave=False)
        done = 0
        try:
            while done < procs:
                try:
                    kind, value = progress_q.get(timeout=0.5)
                except Empty:
                    continue
                if kind == PROG_ADD:
                    doc_bar.update(int(value))
                elif kind == PROG_FILE:
                    file_bar.update(1)
                elif kind == PROG_DONE:
                    done += 1
        finally:
            doc_bar.close()
            file_bar.close()
    t = Thread(target=_runner, daemon=True)
    t.start()
    return t



# ------------------------------
# Args / CLI
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser("Parquet → OpenSearch (multiprocessing)")
    p.add_argument("--path", required=True, help="Parquet file or directory")
    p.add_argument("--recursive", action="store_true", help="Recurse subdirs for *.parquet")
    p.add_argument("--index", required=False, default=os.getenv("OPENSEARCH_INDEX"))
    p.add_argument("--procs", type=int, default=32, help="Worker processes")
    p.add_argument("--chunk_size", type=int, default=8000, help="Rows per Arrow batch (read size)")
    p.add_argument("--bulk_chunk", type=int, default=1000, help="Docs per bulk request (send size)")
    p.add_argument("--max_chunk_bytes", type=int, default=10 * 1024 * 1024, help="Max bytes per bulk request")
    p.add_argument("--pool_maxsize", type=int, default=8, help="per-process HTTP pool sockets")
    p.add_argument("--max_retries", type=int, default=32, help="Transport retries (client-level)")
    p.add_argument("--use_orjson", action="store_true", help="Use orjson serializer (handles NumPy)")
    p.add_argument("--debug", action="store_true", help="Verbose stack traces on connection errors")
    p.add_argument("--error_log", type=str, help="Write critical errors (JSONL) per process with suffix .<pid>.jsonl")
    p.add_argument("--print_retryable", action="store_true", help="Also print retryable errors (noisy)")
    return p.parse_args()


def _run_worker(args_tuple):
    # args_tuple is exactly the tuple you were building in `work`
    return worker_proc(*args_tuple)

# ------------------------------
# Utilities
# ------------------------------
def list_parquet_files(path: str, recursive: bool) -> List[Path]:
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() != ".parquet":
            raise ValueError(f"Not a .parquet file: {p}")
        return [p]
    if not p.is_dir():
        raise ValueError(f"Path not found: {p}")
    pattern = "**/*.parquet" if recursive else "*.parquet"
    files = sorted(p.glob(pattern))
    if not files:
        raise ValueError(f"No .parquet files under {p}")
    return files


def split_round_robin(files: List[Path], k: int) -> List[List[Path]]:
    if k <= 1:
        return [files]
    return [files[i::k] for i in range(k)]


def get_client(max_retries: int, pool_maxsize: int, use_orjson: bool) -> OpenSearch:
    hosts = json.loads(os.getenv("OPENSEARCH_HOSTS", '[{"host":"localhost","port":9200,"scheme":"http"}]'))
    user, pwd = os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASS")
    http_auth = (user, pwd) if user and pwd else None

    kwargs = dict(
        hosts=hosts,
        http_auth=http_auth,
        http_compress=True,
        retry_on_timeout=True,
        verify_certs=False,
        max_retries=max_retries,
        timeout=60,
        connection_class=Urllib3HttpConnection,
        pool_maxsize=pool_maxsize,
    )
    if use_orjson and ORJSONSerializer:
        kwargs["serializer"] = ORJSONSerializer()
    return OpenSearch(**kwargs)


def check_connectivity(client: OpenSearch, debug: bool):
    # DNS preflight
    try:
        hosts = json.loads(os.getenv("OPENSEARCH_HOSTS", "[]"))
        for h in hosts:
            host = h.get("host")
            if not host:
                continue
            socket.getaddrinfo(host, None)
    except Exception as e:
        print(f"❌ DNS resolution failed: {e}", file=sys.stderr)
        if debug:
            traceback.print_exc()
        sys.exit(2)
    # ping + info
    try:
        if not client.ping(params={"request_timeout": 5}):
            raise RuntimeError("Ping returned False (URL/creds/cluster?)")
        _ = client.info()
    except Exception as e:
        print(f"❌ Connection check failed: {e}", file=sys.stderr)
        if debug:
            traceback.print_exc()
        sys.exit(2)


# ------------------------------
# Error classification
# ------------------------------
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
RETRYABLE_TYPES = {
    "es_rejected_execution_exception",
    "cluster_block_exception",
    "master_not_discovered_exception",
    "too_many_requests",
    "receive_timeout_transport_exception",
    "process_cluster_event_timeout_exception",
    "timeout_exception",
}

def classify_bulk_error(item: Dict[str, Any]) -> str:
    op, detail = next(iter(item.items()))
    status = detail.get("status")
    err = detail.get("error") or {}
    et = err.get("type")
    if status in RETRYABLE_STATUS or (et and et in RETRYABLE_TYPES):
        return "retryable"
    return "critical"


def print_retryable(item, pid):
    op, d = next(iter(item.items()))
    status = d.get("status")
    err = d.get("error") or {}
    rid = d.get("_id", "")
    reason = (err.get("reason") or "").splitlines()[0]
    print(f"[{pid}] [retryable] {op} status={status} _id={rid} :: {reason}", file=sys.stderr, flush=True)


def print_critical(item, pid):
    op, d = next(iter(item.items()))
    status = d.get("status")
    err = d.get("error") or {}
    rid = d.get("_id", "")
    reason = (err.get("reason") or "").splitlines()[0]
    et = err.get("type", "unknown_error")
    print(f"[{pid}] [CRITICAL] {op} status={status} type={et} _id={rid} :: {reason}", file=sys.stderr, flush=True)


# ------------------------------
# Batch → actions (fast path for FixedSizeList<float>)
# ------------------------------
def actions_from_batch_numpy(batch: pa.RecordBatch, index_name: str, numpy_ok: bool) -> List[Dict[str, Any]]:
    names = batch.schema.names
    i_chunk = names.index("chunk_id")
    i_emb = names.index("embeddings")
    n = batch.num_rows

    chunk_ids = batch.column(i_chunk).to_pylist()
    emb_field = batch.schema.field(i_emb)
    emb_col = batch.column(i_emb)
    emb_type = emb_field.type

    out: List[Dict[str, Any]] = []

    if pa.types.is_fixed_size_list(emb_type) and pa.types.is_floating(emb_type.value_type):
        dim = emb_type.list_size
        vals = emb_col.values.to_numpy(zero_copy_only=False)  # len n*dim
        mat = vals.reshape(n, dim)
        if numpy_ok and ORJSONSerializer:
            for cid, vec in zip(chunk_ids, mat):
                if cid is None: continue
                out.append({"_index": index_name, "_id": cid, "chunk_id": cid, "embeddings": vec})
        else:
            vecs = mat.tolist()
            for cid, vec in zip(chunk_ids, vecs):
                if cid is None: continue
                out.append({"_index": index_name, "_id": cid, "chunk_id": cid, "embeddings": vec})
        return out

    # Fallback (variable-length lists)
    emb_list = emb_col.to_pylist()
    for cid, vec in zip(chunk_ids, emb_list):
        if cid is None: continue
        out.append({"_index": index_name, "_id": cid, "chunk_id": cid, "embeddings": (np.asarray(vec) if numpy_ok and ORJSONSerializer else vec)})
    return out


def fake_streaming_bulk(
    client: Any,
    actions: Any,
    chunk_size: int = 500,
    max_chunk_bytes: int = 100 * 1024 * 1024,
    raise_on_error: bool = True,
    raise_on_exception: bool = True,
    max_retries: int = 0,
    initial_backoff: int = 2,
    max_backoff: int = 600,
    yield_ok: bool = True,
    ignore_status: Any = (),
    *args: Any,
    **kwargs: Any,
) -> Any:
    for a in actions:
        yield True, a

# ------------------------------
# Worker process
# ------------------------------
def worker_proc(shard_id: int,
                files: List[str],
                index_name: str,
                chunk_rows: int,
                bulk_chunk: int,
                max_chunk_bytes: int,
                pool_maxsize: int,
                max_retries: int,
                use_orjson: bool,
                print_retryable_flag: bool,
                error_log_path: str | None,
                progress_q,    
                progress_tick: int = 1000 ) -> Dict[str, Any]:
    pid = os.getpid()
    stats = defaultdict(int)
    critical_sig_counter: Counter = Counter()

    # per-process error log file (avoid contention)
    log_fp = None
    if error_log_path:
        log_fp = open(f"{error_log_path}.{pid}.jsonl", "w", encoding="utf-8")

    client = get_client(max_retries=max_retries, pool_maxsize=pool_maxsize, use_orjson=use_orjson)

    wanted_cols = ["chunk_id", "embeddings"]
    start = time.time()
    try:
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=chunk_rows, use_threads=True, columns=wanted_cols):
                actions = actions_from_batch_numpy(batch, index_name, numpy_ok=use_orjson)
                sent_since_last = 0
                # stream this batch with streaming_bulk (single process = 1 inflight request)
                for ok, item in streaming_bulk(
                    client,
                    (a for a in actions),
                    chunk_size=bulk_chunk,
                    max_chunk_bytes=max_chunk_bytes,
                    raise_on_error=False,
                ):
                    if ok:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
                        cls = classify_bulk_error(item)
                        if cls == "retryable":
                            stats["retryable"] += 1
                            if print_retryable_flag:
                                print_retryable(item, pid)
                        else:
                            stats["critical"] += 1
                            # count signature
                            op, d = next(iter(item.items()))
                            status = d.get("status", -1)
                            err = d.get("error") or {}
                            et = err.get("type", "unknown_error")
                            reason = (err.get("reason") or "").splitlines()[0][:200]
                            critical_sig_counter[(et, status, reason)] += 1
                            print_critical(item, pid)
                            if log_fp:
                                log_fp.write(json.dumps(item, ensure_ascii=False) + "\n")
                    # progress heartbeat every N docs
                    sent_since_last += 1
                    if sent_since_last >= progress_tick:
                      try:
                        progress_q.put_nowait((PROG_ADD, sent_since_last))
                      except Exception:
                        pass
                      sent_since_last = 0
                # flush any remainder for this batch
                if sent_since_last:
                  try:
                     progress_q.put_nowait((PROG_ADD, sent_since_last))
                  except Exception:
                     pass

    except Exception as e:
        # propagate a clear failure line to parent via stats
        stats["exception"] = 1
        stats["exception_msg"] = str(e)
        stats["exception_tb"] = traceback.format_exc()
    finally:
        if log_fp:
            log_fp.close()
        stats["elapsed"] = time.time() - start
        stats["pid"] = pid
        stats["shard"] = shard_id
        stats["files"] = len(files)
        stats["crit_sigs"] = dict(critical_sig_counter)
        # finished a file
        try:
            progress_q.put_nowait((PROG_FILE, 1))
        except Exception:
            pass
    return stats


# ------------------------------
# Main
# ------------------------------
def main():
    load_dotenv()

    args = parse_args()
    if not args.index:
        print("Missing --index or OPENSEARCH_INDEX", file=sys.stderr); sys.exit(1)

    files = list_parquet_files(args.path, args.recursive)
    total_rows = 0
    for f in files:
        try:
            pf = pq.ParquetFile(str(f))
            total_rows += pf.metadata.num_rows
        except Exception:
            pass

    manager = mp.Manager()
    progress_q = manager.Queue(maxsize=10000)

    # start the background consumer (2 progress bars)
    progress_thread = start_progress_consumer(
        progress_q,
        total_docs=total_rows,        # from parquet metadata; 0 if unknown
        total_files=len(files),
        procs=args.procs
    )

    # quick one-time connectivity check in parent
    parent_client = get_client(args.max_retries, args.pool_maxsize, args.use_orjson)
    check_connectivity(parent_client, args.debug)

    shards = split_round_robin(files, args.procs)
    work: List[Tuple] = []
    for sid, shard in enumerate(shards):
        work.append((
            sid,
            [str(p) for p in shard],
            args.index,
            args.chunk_size,
            args.bulk_chunk,
            args.max_chunk_bytes,
            args.pool_maxsize,
            args.max_retries,
            args.use_orjson,
            args.print_retryable,
            args.error_log,
            progress_q,
            1000
        ))

    print(f"Starting {len(work)} processes over {len(files)} files; "
          f"batch={args.chunk_size} bulk_chunk={args.bulk_chunk} max_chunk_bytes={args.max_chunk_bytes}", flush=True)

    start = time.time()
    agg = defaultdict(int)
    crit_counter = Counter()

    # Use spawn for safety across platforms
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.procs) as pool:
    #mp.set_start_method("spawn", force=True)
    #with mp.Pool(processes=len(work)) as pool:
        # tqdm over processes finishing
        for stats in tqdm(pool.imap_unordered(_run_worker, work, chunksize=1), total=len(work), desc="Workers", position=2):
            # accumulate
            for k, v in stats.items():
                if k == "crit_sigs":
                    crit_counter.update({tuple(sig): count for sig, count in v.items()})
                elif k in ("exception_msg", "exception_tb"):  # printed below if exists
                    continue
                elif isinstance(v, (int, float)):
                    agg[k] += v
            if stats.get("exception", 0):
                print(f"\n[worker {stats.get('pid')}] FAILED:\n{stats.get('exception_msg')}\n{stats.get('exception_tb')}", file=sys.stderr)

    elapsed = time.time() - start
    progress_thread.join(timeout=1)

    # Summary
    print("\nIndexing finished.")
    print(f"Files processed:           {len(files)}")
    if total_rows:
        print(f"Approx rows (metadata):    {total_rows}")
    print(f"Processes:                 {args.procs}")
    print(f"Successfully indexed:      {agg.get('success',0)}")
    print(f"Failed:                    {agg.get('failed',0)}")
    print(f"  • retryable (suppressed):{agg.get('retryable',0)}")
    print(f"  • critical (shown):      {sum(crit_counter.values())}")
    print(f"Total time:                {elapsed:.2f}s")

    if crit_counter:
        print("\nTop critical error signatures:")
        for (et, status, reason), c in crit_counter.most_common(10):
            print(f"  [{status}] {et}  x{c}  :: {reason}")


if __name__ == "__main__":
    main()


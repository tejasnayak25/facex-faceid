"""Benchmark scaling vector search for large rosters (1K, 10K, 100K).
Compares brute-force numpy dot search vs FAISS exact cosine (IndexFlatIP).

Usage:
    python tools/benchmark_scale.py
"""
import time
import numpy as np

def run_numpy_benchmark(num_students=12500, dim=128, num_queries=200):
    print(f"\n--- Numpy Contiguous Matrix Benchmark ---")
    print(f"Roster size: {num_students:,} | Embedding dim: {dim} | Queries: {num_queries}")
    
    # Pre-normalized database
    db = np.random.randn(num_students, dim).astype(np.float32)
    db /= (np.linalg.norm(db, axis=1, keepdims=True) + 1e-9)

    queries = np.random.randn(num_queries, dim).astype(np.float32)
    queries /= (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)

    # Warmup
    _ = np.dot(db, queries[0])

    t0 = time.perf_counter()
    for q in queries:
        scores = np.dot(db, q)
        best_idx = np.argmax(scores)
    elapsed = (time.perf_counter() - t0) * 1000

    per_query_ms = elapsed / num_queries
    qps = num_queries / (elapsed / 1000)
    print(f"Total time: {elapsed:.2f} ms")
    print(f"Per-query latency: {per_query_ms:.4f} ms")
    print(f"Throughput: {qps:.1f} queries/sec")

def run_faiss_benchmark(num_students=100000, dim=128, num_queries=1000):
    try:
        import faiss
    except ImportError:
        print("\n[Optional] FAISS not installed. Run `pip install faiss-cpu` to test FAISS 100K scaling.")
        return

    print(f"\n--- FAISS IndexFlatIP (Exact Cosine) Benchmark ---")
    print(f"Roster size: {num_students:,} | Embedding dim: {dim} | Queries: {num_queries}")

    db = np.random.randn(num_students, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)
    faiss.normalize_L2(db)
    faiss.normalize_L2(queries)

    index = faiss.IndexFlatIP(dim)
    index.add(db)

    # Warmup
    index.search(queries[:5], k=1)

    t0 = time.perf_counter()
    scores, indices = index.search(queries, k=1)
    elapsed = (time.perf_counter() - t0) * 1000

    per_query_ms = elapsed / num_queries
    qps = num_queries / (elapsed / 1000)
    print(f"Total time: {elapsed:.2f} ms")
    print(f"Per-query latency: {per_query_ms:.4f} ms")
    print(f"Throughput: {qps:.1f} queries/sec")

if __name__ == "__main__":
    run_numpy_benchmark(12500, 128, 500)
    run_numpy_benchmark(100000, 128, 200)
    run_faiss_benchmark(100000, 128, 1000)

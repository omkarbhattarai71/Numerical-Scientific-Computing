import time
from mandelbrot_common import default_params
from dask.distributed import Client, LocalCluster
import mandelbrot_dask

def run_benchmarks_distributed():
    """Run quick benchmark at 4096x4096 with Dask distributed client."""
    
    # Create a local cluster (scheduler + workers) in this script
    print("Starting Dask LocalCluster...")
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, processes=True)
    client = Client(cluster)
    print("Connected to Dask Cluster")
    print(client)
    
    width, height = 4096, 4096
    print(f"\n--- Benchmarking Resolution: {width}x{height} ---")
    
    params = default_params()
    params['width'] = width
    params['height'] = height
    
    # Run Dask with (256, 256) chunks
    chunk_size = (256, 256)
    print(f"\nRunning Dask with chunk size {chunk_size}...")
    start = time.time()
    mandelbrot_dask.mandelbrot_dask(params, chunk_size=chunk_size, client=client)
    dask_time = time.time() - start
    print(f"\n=== Dask Distributed (Chunk Size: {chunk_size}): {dask_time:.2f}s ===")
    
    client.close()
    cluster.close()
    print("\nDask cluster closed.")

if __name__ == "__main__":
    run_benchmarks_distributed()

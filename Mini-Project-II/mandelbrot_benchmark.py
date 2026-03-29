import time
import csv
import multiprocessing as mp
from mandelbrot_common import default_params
# Assume the below are imported from your respective files
import mandelbrot_numpy
import mandelbrot_multiproc
import mandelbrot_dask

def run_benchmarks():
    results = []
    
    # Scaling Analysis: Start small -> increase gradually
    # We increase resolution to force longer execution times
    resolutions = [(1024, 1024), (2048, 2048), (4096, 4096)] 
    
    # Multiprocessing parameters to test
    max_cores = mp.cpu_count()
    core_counts = [1, max_cores // 2, max_cores]
    mp_chunk_multipliers = [1, 4, 8] # chunk = cores * multiplier
    
    # Dask parameters to test
    dask_chunk_sizes = [(128, 128), (256, 256), (512, 512)]

    for width, height in resolutions:
        print(f"\n--- Benchmarking Resolution: {width}x{height} ---")
        params = default_params()
        params['width'] = width
        params['height'] = height
        
        # 1. Baseline NumPy
        start = time.time()
        mandelbrot_numpy.mandelbrot_numpy(**params)
        np_time = time.time() - start
        results.append({'Method': 'NumPy Vectorized', 'Resolution': f"{width}x{height}", 
                        'Cores/Workers': 1, 'Chunk Size': 'N/A', 'Time (s)': np_time, 'Speedup': 1.0})
        print(f"NumPy Vectorized: {np_time:.2f}s")

        # 2. Multiprocessing
        for cores in core_counts:
            for mult in mp_chunk_multipliers:
                chunks = cores * mult
                start = time.time()
                mandelbrot_multiproc.mandelbrot_multiproc(**params, workers=cores, chunks=chunks)
                mp_time = time.time() - start
                results.append({'Method': 'Multiprocessing', 'Resolution': f"{width}x{height}",'Cores/Workers': cores, 
                                'Chunk Size': f"{chunks} total chunks", 'Time (s)': mp_time, 'Speedup': np_time / mp_time})
                print(f"Multiprocessing (Cores: {cores}, Chunks: {chunks}): {mp_time:.2f}s")

        # 3. Dask (Local Multi-core)
        for d_chunk in dask_chunk_sizes:
            start = time.time()
            mandelbrot_dask.mandelbrot_dask(**params, chunk_size=d_chunk) # Assuming local client initiated inside
            dask_time = time.time() - start
            results.append({'Method': 'Dask Local', 'Resolution': f"{width}x{height}",'Cores/Workers': max_cores, 'Chunk Size': f"{d_chunk}", 
                            'Time (s)': dask_time, 'Speedup': np_time / dask_time})
            print(f"Dask Local (Chunk Size: {d_chunk}): {dask_time:.2f}s")

    # Save to CSV
    csv_filename = "mandelbrot_timing_results.csv"
    keys = results[0].keys()
    with open(csv_filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    print(f"\nBenchmarking complete. Results saved to {csv_filename}")

if __name__ == "__main__":
    run_benchmarks()
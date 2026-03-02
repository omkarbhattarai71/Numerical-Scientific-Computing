# bench.py
import time
from mandelbrot_common import default_params
from mandelbrot_baseline import mandelbrot_baseline
from mandelbrot_numpy import mandelbrot_numpy
from mandelbrot_numba import mandelbrot_numba
from mandelbrot_multiproc import mandelbrot_multiproc

def run_and_time(fn, label, params):
    t0 = time.perf_counter()
    itc = fn(**params)
    dt = time.perf_counter() - t0
    print(f"{label:22s}  {dt:.3f}s   shape={itc.shape}")
    return itc

if __name__ == "__main__":
    params = default_params()  
    run_and_time(mandelbrot_baseline,   "Baseline", params)
    run_and_time(mandelbrot_numpy,      "NumPy", params)
    run_and_time(mandelbrot_numba,      "Numba JIT", params)
    run_and_time(mandelbrot_multiproc,  "Multiprocessing", params)
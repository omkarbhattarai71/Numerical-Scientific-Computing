Readme Please :)

## Dependencies

Ensure you have the necessary Python libraries installed before running the scripts:

```bash
pip install numpy matplotlib dask distributed
```

## Execution Instructions


```bash
python3 mandelbrot_numpy.py
python3 mandelbrot_multiproc.py
python3 mandelbrot_dask.py
```

### 2. Running Local Benchmarks


```bash
python3 mandelbrot_benchmark.py
```

### 3. Running Distributed Dask Benchmarks
*Note: In order to run the code: `mandelbrot_benchmark_distributed.py`, three terminal should be opened, following should be run.*

* **Terminal 1:** `dask-scheduler`
* **Terminal 2:** `dask-worker tcp://127.0.0.1:8786`
* **Terminal 3:** `python3 mandelbrot_benchmark_distributed.py`

## Outputs
Running the benchmark scripts - `mandelbort_benchmark.py` will output execution times to the console and save the data to `mandelbrot_timing_results.csv` for speedup comparison and analysis. Running the individual scripts will output high-resolution `.png` images of the Mandelbrot fractal.



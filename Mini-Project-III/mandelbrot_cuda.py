import numpy as np
from numba import cuda
import math
import unittest
import time
import csv
import warnings
from numba.core.errors import NumbaPerformanceWarning

import matplotlib
matplotlib.use('Agg') # Headless mode for cluster execution
import matplotlib.pyplot as plt

# ==========================================
# 1. NUMPY IMPLEMENTATION
# ==========================================
def mandelbrot_numpy(x_min, x_max, y_min, y_max, width, height, max_iter):
    """
    Compute the Mandelbrot set using NumPy vectorized operations.

    Parameters:
        x_min (float): Minimum real value
        x_max (float): Maximum real value
        y_min (float): Minimum imaginary value
        y_max (float): Maximum imaginary value
        width (int): Grid width
        height (int): Grid height
        max_iter (int): Maximum iterations

    Returns:
        np.ndarray: 2D array of iteration counts
    """
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    C = X + 1j * Y
    Z = np.zeros_like(C)

    output = np.zeros(C.shape, dtype=np.int32)
    mask = np.ones(C.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask]**2 + C[mask]
        escaped = np.abs(Z[mask]) > 2.0

        just_escaped = mask.copy()
        just_escaped[mask] = escaped
        output[just_escaped] = i

        mask[mask] = ~escaped

        if not np.any(mask):
            break

    output[mask] = max_iter
    return output


# ==========================================
# 2. CUDA IMPLEMENTATION & REDUCTION
# ==========================================
@cuda.jit
def mandelbrot_kernel(d_output, x_min, x_max, y_min, y_max, width, height, max_iter):
    """
    CUDA kernel computing Mandelbrot set.
    Each thread computes one pixel.
    """
    x, y = cuda.grid(2)

    if x < width and y < height:
        real = x_min + (x / width) * (x_max - x_min)
        imag = y_min + (y / height) * (y_max - y_min)

        c = complex(real, imag)
        z = 0.0j

        for i in range(max_iter):
            z = z * z + c

            if (z.real * z.real + z.imag * z.imag) > 4.0:
                d_output[y, x] = i
                return

        d_output[y, x] = max_iter

@cuda.reduce
def sum_reduction(a, b):
    """
    Bonus Feature: CUDA Reduction kernel to sum all iterations.
    """
    return a + b

# ==========================================
# 3. BENCHMARK FUNCTIONS
# ==========================================
def run_numpy(width, height, max_iter):
    """Measure NumPy execution time."""
    start = time.time()
    mandelbrot_numpy(-2, 1, -1.5, 1.5, width, height, max_iter)
    return time.time() - start

def run_cuda(width, height, max_iter, block_size):
    """Run CUDA kernel and measure precise GPU execution time."""
    d_out = cuda.device_array((height, width), dtype=np.int32)

    threads = (block_size, block_size)
    blocks = (
        math.ceil(width / block_size),
        math.ceil(height / block_size)
    )

    # Warm-up (JIT compile)
    mandelbrot_kernel[blocks, threads](d_out, -2, 1, -1.5, 1.5, width, height, max_iter)
    cuda.synchronize()

    start = cuda.event(timing=True)
    end = cuda.event(timing=True)

    start.record()
    mandelbrot_kernel[blocks, threads](d_out, -2, 1, -1.5, 1.5, width, height, max_iter)
    end.record()
    end.synchronize()

    return cuda.event_elapsed_time(start, end) / 1000.0


# ==========================================
# 4. SCALING EXPERIMENT & PLOTTING
# ==========================================
def scaling_experiment():
    """Run scaling experiments and plot the results."""
    sizes = [256, 512, 1024, 2048]
    block_sizes = [8, 16, 32]
    max_iter = 100

    results = []
    numpy_times = []
    cuda_best_times = []

    print("\n=== Scaling Experiment ===")
    for size in sizes:
        print(f"\nGrid: {size}x{size}")
        
        n_time = run_numpy(size, size, max_iter)
        print(f"NumPy: {n_time:.4f}s")
        numpy_times.append(n_time)

        best_c_time = float('inf')
        for b in block_sizes:
            c_time = run_cuda(size, size, max_iter, b)
            speedup = n_time / c_time
            print(f"CUDA block {b}: {c_time:.4f}s | Speedup: {speedup:.2f}x")
            
            results.append([size, b, n_time, c_time, speedup])
            if c_time < best_c_time:
                best_c_time = c_time
                
        cuda_best_times.append(best_c_time)

    # Save CSV
    with open("timing_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "block_size", "numpy_time", "cuda_time", "speedup"])
        writer.writerows(results)
    print("\nSaved timing_results.csv")

    # Generate Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, numpy_times, marker='o', label='NumPy (CPU Vectorized)', linewidth=2)
    plt.plot(sizes, cuda_best_times, marker='s', label='CUDA (Best Block Size)', linewidth=2)
    plt.title('Mandelbrot Execution Time: NumPy vs. CUDA')
    plt.xlabel('Grid Resolution (N x N)')
    plt.ylabel('Execution Time (Seconds)')
    plt.yscale('log') # Log scale is best to show the massive gap
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('scaling_plot.png', dpi=300)
    plt.close()
    print("Saved scaling_plot.png")


# ==========================================
# 5. IMAGE GENERATION & REDUCTION
# ==========================================
def save_image():
    """Generate image and calculate mean iterations using CUDA reduction."""
    width, height = 1024, 1024
    max_iter = 100

    d_out = cuda.device_array((height, width), dtype=np.int32)
    threads = (16, 16)
    blocks = (math.ceil(width/16), math.ceil(height/16))

    mandelbrot_kernel[blocks, threads](d_out, -2, 1, -1.5, 1.5, width, height, max_iter)
    
    # BONUS: Calculate Mean Iteration using Shared Memory Reduction
    # Flatten the array for the reduction algorithm
    total_iters = sum_reduction(d_out.ravel())
    mean_iters = total_iters / (width * height)
    print(f"\nBONUS: Mean iterations per pixel (calculated via CUDA reduction): {mean_iters:.2f}")

    img = d_out.copy_to_host()

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='hot')
    plt.colorbar()
    plt.title("Mandelbrot Set (CUDA)")
    plt.savefig("mandelbrot_final.png", dpi=300)
    plt.close()
    print("Saved mandelbrot_final.png")


# ==========================================
# 6. UNIT TESTS
# ==========================================
class TestMandelbrot(unittest.TestCase):
    def test_shape(self):
        result = mandelbrot_numpy(-2, 1, -1.5, 1.5, 100, 100, 50)
        self.assertEqual(result.shape, (100, 100))

    def test_divergent_point(self):
        d_out = cuda.device_array((1, 1), dtype=np.int32)
        mandelbrot_kernel[(1, 1), (1, 1)](d_out, 2, 2, 0, 0, 1, 1, 50)
        self.assertLess(d_out.copy_to_host()[0, 0], 50)

    def test_stable_point(self):
        d_out = cuda.device_array((1, 1), dtype=np.int32)
        mandelbrot_kernel[(1, 1), (1, 1)](d_out, 0, 0, 0, 0, 1, 1, 50)
        self.assertEqual(d_out.copy_to_host()[0, 0], 50)


# ==========================================
# 7. MAIN
# ==========================================
if __name__ == "__main__":

    print("Running unit tests...")
    # Catch warnings to hide the expected 1x1 grid Numba under-utilization warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NumbaPerformanceWarning)
        unittest.main(argv=[''], exit=False)

    scaling_experiment()
    save_image()
    print("\nDONE.")

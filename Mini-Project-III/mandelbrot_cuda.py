import numpy as np
from numba import cuda
import math
import unittest
import time

# ==========================================
# 1. UPDATED PREVIOUS CODE & DOCSTRINGS
# ==========================================

def mandelbrot_numpy(x_min: float, x_max: float, y_min: float, y_max: float, 
                     width: int, height: int, max_iter: int) -> np.ndarray:
    """
    Computes the Mandelbrot set using vectorized NumPy operations.
    
    This function generates a 2D grid of complex numbers and iteratively 
    applies the Mandelbrot formula (Z = Z^2 + C). It uses boolean masking 
    to track which points have escaped the threshold radius of 2.0.
    
    Args:
        x_min (float): Minimum real value (x-axis).
        x_max (float): Maximum real value (x-axis).
        y_min (float): Minimum imaginary value (y-axis).
        y_max (float): Maximum imaginary value (y-axis).
        width (int): The number of pixels in the x-direction.
        height (int): The number of pixels in the y-direction.
        max_iter (int): The maximum number of iterations before assuming the point is in the set.
        
    Returns:
        np.ndarray: A 2D array of shape (height, width) containing the escape iteration count for each pixel.
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
        
        # Update output for pixels that just escaped
        just_escaped_mask = mask.copy()
        just_escaped_mask[mask] = escaped
        output[just_escaped_mask] = i
        
        # Remove escaped pixels from the active mask
        mask[mask] = ~escaped
        
        if not np.any(mask):
            break
            
    output[mask] = max_iter
    return output

# ==========================================
# 2. CUDA IMPLEMENTATION & BONUS REDUCTION
# ==========================================

@cuda.jit
def mandelbrot_kernel(d_output, x_min, x_max, y_min, y_max, width, height, max_iter):
    """
    CUDA kernel to compute the Mandelbrot set.
    
    Each thread calculates the escape time for a single pixel in the output image.
    It maps the 2D thread indices (x, y) to the complex plane coordinates.
    
    Args:
        d_output (DeviceNDArray): The output array allocated on the GPU.
        x_min, x_max, y_min, y_max (float): Complex plane boundaries.
        width, height (int): Grid dimensions.
        max_iter (int): Maximum iteration threshold.
    """
    # 2D Grid/Block configuration calculation
    x, y = cuda.grid(2)
    
    # Out-of-bounds check (Guard)
    if x < width and y < height:
        # Map pixel to complex plane
        real = x_min + (x / width) * (x_max - x_min)
        imag = y_min + (y / height) * (y_max - y_min)
        
        c = complex(real, imag)
        z = 0.0j
        
        for i in range(max_iter):
            z = z**2 + c
            if (z.real * z.real + z.imag * z.imag) > 4.0: # Math equivalent of abs(z) > 2.0
                d_output[y, x] = i
                return
                
        d_output[y, x] = max_iter

@cuda.reduce
def sum_reduction(a, b):
    """
    Bonus Feature: CUDA Reduction kernel to sum all iterations.
    Used to calculate the mean iteration count across the image.
    """
    return a + b

# ==========================================
# 3. UNIT TESTING (unittest)
# ==========================================

class TestMandelbrot(unittest.TestCase):
    
    def setUp(self):
        self.x_min, self.x_max = -2.0, 1.0
        self.y_min, self.y_max = -1.5, 1.5
        self.width, self.height = 100, 100
        self.max_iter = 50
        
    def test_numpy_shape(self):
        """Test Case 1: Ensure output shape is correct."""
        result = mandelbrot_numpy(self.x_min, self.x_max, self.y_min, self.y_max, 
                                  self.width, self.height, self.max_iter)
        self.assertEqual(result.shape, (self.height, self.width))
        
    def test_cuda_known_divergent_point(self):
        """Test Case 2: Ensure a known divergent point (c=2+0j) escapes quickly."""
        # Create a tiny 1x1 grid focusing on x=2.0, y=0.0
        d_out = cuda.device_array((1, 1), dtype=np.int32)
        threads_per_block = (1, 1)
        blocks_per_grid = (1, 1)
        
        mandelbrot_kernel[blocks_per_grid, threads_per_block](
            d_out, 2.0, 2.0, 0.0, 0.0, 1, 1, self.max_iter)
        
        result = d_out.copy_to_host()
        self.assertLess(result[0, 0], self.max_iter) # Should escape before max_iter
        
    def test_cuda_known_stable_point(self):
        """Test Case 3: Ensure a known stable point (c=0+0j) reaches max_iter."""
        d_out = cuda.device_array((1, 1), dtype=np.int32)
        threads_per_block = (1, 1)
        blocks_per_grid = (1, 1)
        
        mandelbrot_kernel[blocks_per_grid, threads_per_block](
            d_out, 0.0, 0.0, 0.0, 0.0, 1, 1, self.max_iter)
        
        result = d_out.copy_to_host()
        self.assertEqual(result[0, 0], self.max_iter)

if __name__ == '__main__':
    # Run tests first
    print("Running Unit Tests...")
    unittest.main(exit=False)
    print("\nTests completed. To run the full benchmark, execute the worksheet code separately.")
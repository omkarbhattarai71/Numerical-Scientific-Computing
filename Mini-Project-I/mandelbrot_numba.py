import time
from mandelbrot_common import complex_grid, normalize_for_plot, default_params
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange

@jit(nopython=True, parallel=True)
def _mandelbrot_numba_kernel(xmin, xmax, ymin, ymax, width, height, max_iter):
    it_counts = np.zeros((height, width), dtype=np.uint16)
    # Precompute the steps to avoid the repeated division
    dx = (xmax - xmin) / (width - 1)
    dy = (ymax - ymin) / (height - 1)

    for j in prange(height):    # prange enables multi-core paraller execution of loops
        y = ymin + j * dy
        for i in range(width):
            x = xmin + i * dx
            c_re, c_im = x, y
            z_re, z_im = 0.0, 0.0
            count = 0
            while count < max_iter and (z_re*z_re + z_im*z_im) <= 4.0:
                z_re, z_im = (z_re*z_re - z_im*z_im + c_re), (2.0*z_re*z_im + c_im)
                count += 1
            it_counts[j, i] = count
    return it_counts

def mandelbrot_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    return _mandelbrot_numba_kernel(xmin, xmax, ymin, ymax, width, height, max_iter)    

if __name__ == "__main__":  
    starttime = time.time()
    params = default_params()
    itc = mandelbrot_numba(**params) # Compute the Mandelbrot set using the optimized Numba version
    img = normalize_for_plot(itc, params["max_iter"])
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="turbo", extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]])
    plt.title("Mandelbrot (Numba)")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.tight_layout()
    plt.savefig('mandelbrot_numba.png', dpi=300, bbox_inches='tight')
    print("Plot saved as mandelbrot_numba.png")
    plt.show()
    endtime = time.time()
    print(f"Execution time: {endtime - starttime:.2f} seconds")
    
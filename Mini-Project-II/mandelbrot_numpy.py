import time
from mandelbrot_common import complex_grid, normalize_for_plot, default_params
import numpy as np
import matplotlib.pyplot as plt
def mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter=100):  
    C, _, _ = complex_grid(xmin, xmax, ymin, ymax, width, height)
    Z = np.zeros_like(C, dtype=np.complex128)
    it_counts = np.zeros(C.shape, dtype=np.uint16)
    mask = np.ones(C.shape, dtype=bool)  # Mask to track points that are still in the set

    for k in range(1, max_iter + 1):
        Z[mask] = Z[mask] * Z[mask] + C[mask]  # Update Z for points still in the set
        escaped_now = (Z.real*Z.real + Z.imag*Z.imag) > 4.0  # Check which points have escaped
        newly_escaped = escaped_now & mask
        it_counts[newly_escaped] = k  # Set iteration count for newly escaped points
        mask &= ~escaped_now  # Update mask to exclude points that have escaped
        if not mask.any():  # If all points have escaped, we can stop early
            break
    it_counts[mask] = max_iter  # Set iteration count for points that never escaped
    return it_counts

if __name__ == "__main__":
    starttime = time.time()
    params = default_params()
    itc = mandelbrot_numpy(**params) # Compute the Mandelbrot set using the optimized Numpy version
    img = normalize_for_plot(itc, params["max_iter"])
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="turbo", extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]])
    plt.title("Mandelbrot (Numpy)")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.tight_layout()
    plt.savefig('mandelbrot_numpy.png', dpi=300, bbox_inches='tight')
    print("Plot saved as mandelbrot_numpy.png")
    plt.show()
    endtime = time.time()
    print(f"Execution time: {endtime - starttime:.2f} seconds")
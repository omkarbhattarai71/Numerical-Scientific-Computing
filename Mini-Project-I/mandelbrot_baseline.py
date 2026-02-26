# mandelbrot_baseline.py
import time

from mandelbrot_common import complex_grid, normalize_for_plot, default_params
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_baseline(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    C, _, _ = complex_grid(xmin, xmax, ymin, ymax, width, height)
    it_counts = np.zeros((height, width), dtype=np.uint16)

    # Pure Python/Numpy-scalar loop: [not fast]
    for j in range(height):
        for i in range(width):
            c = C[j, i]
            z = 0+0j
            count = 0
            # Iterate z_{n+1} = z_n^2 + c up to max_iter
            while count < max_iter and (z.real*z.real + z.imag*z.imag) <= 4.0: # |z|^2 <= 4 equivalent to |z|<=2
                z = z*z + c
                count += 1
            it_counts[j, i] = count
    return it_counts

if __name__ == "__main__":
    starttime = time.time()
    params = default_params()
    itc = mandelbrot_baseline(**params)
    img = normalize_for_plot(itc, params["max_iter"])
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="turbo", extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]])
    plt.title("Mandelbrot (Baseline)")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.tight_layout()
    plt.savefig('mandelbrot_baseline.png', dpi=300, bbox_inches='tight')
    print("Plot saved as mandelbrot_baseline.png")
    plt.show()
    endtime = time.time()
    print(f"Execution time: {endtime - starttime:.2f} seconds")


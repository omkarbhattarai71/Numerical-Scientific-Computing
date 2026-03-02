import time
from mandelbrot_common import complex_grid, normalize_for_plot, default_params
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count     

def _compute_strip(args):
    """ Worker: Compute a horizontal strip of the Mandelbrot set, using local fast scaler loop.  """
    xmin, xmax, ymin, ymax, width, height, max_iter, row_start, row_end = args
    out = np.zeros((row_end - row_start, width), dtype=np.uint16)

    dx = (xmax - xmin) / (width - 1)
    dy = (ymax - ymin) / (height - 1)   
    for j in range(row_start, row_end):
        y = ymin + j * dy
        row = j - row_start
        for i in range(width):
            x = xmin + i * dx
            c_re, c_im = x, y
            z_re, z_im = 0.0, 0.0
            count = 0
            while count < max_iter and (z_re*z_re + z_im*z_im) <= 4.0:
                z_re, z_im = (z_re*z_re - z_im*z_im + c_re), (2.0*z_re*z_im + c_im)
                count += 1
            out[row, i] = count
    return (row_start, out)

def mandelbrot_multiproc(xmin, xmax, ymin, ymax, width, height, max_iter=100, workers=None, chunks=cpu_count()*4):
    """ Compute the Mandelbrot set using multiprocessing.  """
    if workers is None:
        workers = max(1, cpu_count() - 1)  # Use all but one core by default
    # build strips (contiguous row ranges) to keep memroy writes cache-friendly
    indices = np.linspace(0, height, chunks+1, dtype=int)
    tasks = [(xmin, xmax, ymin, ymax, width, height, max_iter, indices[k], indices[k+1]) 
             for k in range(chunks)
             if int(indices[k]) < int(indices[k+1])]  # Filter out empty strips
    
    out = np.zeros((height, width), dtype=np.uint16)
    with Pool(processes=workers) as pool:
        for row_start, block in pool.imap_unordered(_compute_strip, tasks):
            out[row_start:row_start+block.shape[0], :] = block
    return out

if __name__ == "__main__":
    starttime = time.time()
    params = default_params()
    itc = mandelbrot_multiproc(**params) # Compute the Mandelbrot set using the multiprocessing version
    img = normalize_for_plot(itc, params["max_iter"])
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="turbo", extent=[params["xmin"], params["xmax"], params["ymin"], params["ymax"]])
    plt.title("Mandelbrot (Multiprocessing)")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.tight_layout()
    plt.savefig('mandelbrot_multiproc.png', dpi=300, bbox_inches='tight')
    print("Plot saved as mandelbrot_multiproc.png")
    plt.show()
    endtime = time.time()
    print(f"Execution time: {endtime - starttime:.2f} seconds")
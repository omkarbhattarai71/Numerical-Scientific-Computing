# This function will have 3 functions (to reuse the code)
import numpy as np

def complex_grid(xmin=-2.0, xmax=1.0, ymin=-1.5, ymax=1.5, width=1024, height=1024):
    """ Build a complex grid c= X +iY covering the given rectangle in the complex plane.   """
    xs = np.linspace(xmin, xmax, width, dtype=np.float32)
    ys = np.linspace(ymin, ymax, height, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys) # meshgrid is used to create a grid of coordinates from the 1D arrays xs and ys
    C = X + 1j * Y
    return C, xs, ys

def normalize_for_plot(it_counts, max_iter):
    """ Normalize iteration counts for plotting.
     Returns a float array in [0,1] """
    arr = it_counts.astype(np.float32)
    return arr / max_iter

def default_params():
    """ Return default parameters for Mandelbrot set computation. """
    return dict(
        xmin=-2.0,
        xmax=1.0,
        ymin=-1.5,
        ymax=1.5,
        width=1024,
        height=1024,
        max_iter=100
    )
import numpy as np
import dask.array as da
from dask.distributed import Client
import time
from mandelbrot_common import default_params, normalize_for_plot
import matplotlib.pyplot as plt

def compute_mandelbrot_chunk(c_chunk, max_iter):
    """
    Core logic using NumPy arrays. Dask calls this via map_blocks.
    Notes: We use NumPy here, not Dask arrays, for speed.
    """
    z = np.zeros_like(c_chunk, dtype=np.complex64)
    iters = np.zeros(c_chunk.shape, dtype=np.uint16)
    
    for i in range(max_iter):
        mask = np.abs(z) <= 2.0
        if not np.any(mask): 
            break  
            
        z[mask] = z[mask]**2 + c_chunk[mask]
        iters[mask] += 1
        
    return iters

def mandelbrot_dask(params, chunk_size=(256, 256), client=None):
    """
    Implements the Dask version using Dask Arrays and map_blocks.
    """
    # 1. Create the complex grid coordinates
    x = np.linspace(params['xmin'], params['xmax'], params['width'])
    y = np.linspace(params['ymin'], params['ymax'], params['height'])
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    # 2. Convert to Dask Array with specific chunking 
    d_c = da.from_array(C, chunks=chunk_size)

    # 3. Use map_blocks to apply our NumPy function 
    # drop_axis/new_axis aren't needed here as shape remains (H, W)
    d_mandel = d_c.map_blocks(compute_mandelbrot_chunk, max_iter=params['max_iter'], dtype=np.uint16)

    # 4. Compute the result
    return d_mandel.compute()

if __name__ == "__main__":
    # To run on a cluster (Strato or local multi-node), initialize the Client: client = Client('tcp://scheduler-address:8786') 

    client = Client() 
    print(f"Dask Dashboard available at: {client.dashboard_link}")

    params = default_params()
    
    # Scaling Analysis: Try different chunk sizes (e.g., 128, 256, 512)
    print("Starting Dask computation...")
    start = time.time()
    
    result = mandelbrot_dask(params, chunk_size=(256, 256), client=client)
    
    end = time.time()
    print(f"Dask Execution Time: {end - start:.2f} seconds")

    # Visualization
    img = normalize_for_plot(result, params["max_iter"])
    plt.imshow(img, cmap="turbo")
    plt.title("Mandelbrot (Dask Implementation)")
    plt.savefig('mandelbrot_dask.png', dpi=300, bbox_inches='tight')
    print("Plot saved as mandelbrot_dask.png")
    plt.show()

    # Keep the Dask scheduler running for dashboard access
    print("\nDask Dashboard still available at:", client.dashboard_link)
    print("Press Ctrl+C to stop...")
    import signal
    
    def signal_handler(sig, frame):
        print("\n\nShutting down Dask client (this may take a moment)...")
        try:
            client.close(timeout=5)
        except Exception:
            pass
        import sys
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
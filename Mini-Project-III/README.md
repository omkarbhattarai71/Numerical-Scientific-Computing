# Mandelbrot CUDA - Running mandelbrot_cuda.sh

This README explains how to run the mandelbrot_cuda.sh script which builds and/or runs the CUDA implementation of the Mandelbrot renderer included in this repository.

Prerequisites
- Linux with an NVIDIA GPU (Here, AI lab)
- CUDA Toolkit (nvcc) installed and on PATH
- bash shell

Quick start
1. Open a terminal at AI lab and transfer the folder, Mini-Project-III, into it. 

3. cd Mini-Project_III 

2. Run the script mandelbrot_cuda.sh using, *sbatch mandelbrot_cuda.sh*, where it will run *mandelbrot_cuda.py* file that will generate all other files, scaling_plot.png, timing_results.csv, slurm error and output. 

Note: all the necessary libraries are required to be installed prior to running the batch. 


#!/bin/bash
#SBATCH --job-name=mandelbrot_cuda
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

echo "======================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "======================================"

# Activate virtual environment
source ~/num_sci/venv/bin/activate

# Print GPU info to verify allocation
nvidia-smi

echo "--------------------------------------"
echo "Running Mandelbrot CUDA script..."
echo "--------------------------------------"

# Run Python script and use 'tee' to capture output in both the SLURM log AND a separate file
python3 -u mandelbrot_cuda.py 2>&1 | tee python_output.log

echo "--------------------------------------"
echo "Execution finished"
echo "--------------------------------------"

echo "Job finished at: $(date)"

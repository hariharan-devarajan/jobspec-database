#!/bin/bash
#SBATCH --job-name=babel_hipsycl
#SBATCH --account=project
#SBATCH --partition=gpumedium
#SBATCH --time=00:02:00
#SBATCH -e babel_hipsycl_v100_error
#SBATCH -o babel_hipsycl_v100_out
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1


for i in {1..10}; do
        echo $i
        srun -n 1 ./sycl-stream --device 1;
        sleep 5;
done

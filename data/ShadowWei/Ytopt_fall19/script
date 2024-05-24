#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --gres=gpu:p100:1
#SBATCH --job-name=comp_422_openmp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --export=ALL
ulimit -c unlimited -s
#module load impi/2019.2.187
#source /uufs/chpc.utah.edu/common/home/u1074259/ytopt/experiments/exp-6/jobs/000_00030.job                 
#python /uufs/chpc.utah.edu/common/home/u1074259/ytopt/problems/atax/executable.py --p0 a --p1 a --p2 c
mpiexec -n 2 python -m ytopt.search.async_search --prob_path=problems/convolution-2d2/problem.py --exp_dir=experiments/convolution_mk4 --prob_attr=problem --exp_id=convolution_mk4  --max_time=60 --base_estimator='RF' --patience_fac=30
#mpirun -np 2 python -m ytopt.search.async_search --prob_path=problems/correlation/problem.py --exp_dir=experiments/exp-6 --prob_attr=problem --exp_id=exp-6  --max_time=60 --base_estimator='RF'
#mpiexec -n 2 python -m ytopt.search.async_search --prob_path=problems/atax2/problem.py --exp_dir=experiments/atax_mk3 --prob_attr=problem --exp_id=atax_mk3  --max_time=180 --base_estimator='RF'

#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=comp_422_openmp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
lscpu
. ~/spack/share/spack/setup-env.sh
spack load rose
spack load boost
ROSEHOME=`spack location -i rose`
BOOSTHOME=`spack location -i boost`
IEGENHOME=/uufs/chpc.utah.edu/common/home/u1142914/lib/install

export ROSEHOME
export BOOSTHOME
export IEGENHOME

module load isl-0.19-gcc-6.3.0-xn6q7xj


module load anaconda/5.3.0
source activate ytopt
module load gcc/6.4.0
module load mpich


ulimit -c unlimited -s
polybenchDIR=polybench/polybench-code
python --version
python3 --version
which clang
export LD_LIBRARY_PATH=$HOME/lib/openmp-build/runtime/src:$LD_LIBRARY_PATH
set OMP_NUM_THREADS=8

#module load anaconda/5.3.0
#source activate ytopt
#module load gcc/6.4.0
#module load isl-0.19-gcc-6.3.0-xn6q7xj
#module load mpich
#ulimit -c unlimited -s
#python --version
#python3 --version
#which clang

python exp.py


#module load gcc/8.1.0
#cd /uufs/chpc.utah.edu/common/home/u1142914/lib/ytopt_vinu/polybench/polybench-code/utilities
#python run.py

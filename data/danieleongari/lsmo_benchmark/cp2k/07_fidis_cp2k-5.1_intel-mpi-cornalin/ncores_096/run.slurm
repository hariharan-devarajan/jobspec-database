#!/bin/bash -l
#SBATCH --nodes=4
#SBATCH --ntasks=96
#SBATCH --time=00:30:00
#SBATCH --partition=parallel
#SBATCH --constraint=E5v4


source /ssoft/spack/bin/slmodules.sh -r deprecated   
module load intel
module load intel-mpi intel-mkl
module load cp2k/5.1-mpi

date_start=$(date +%s)
srun /ssoft/spack/cornalin/v2/opt/spack/linux-rhel7-x86_E5v4_Mellanox/intel-17.0.2/cp2k-5.1-gke4ujsx5qhx6zt7ncfo22hajqhp6n6b/bin/cp2k.popt -i *.inp -o output.out                 #running command
date_end=$(date +%s)
time_run=$((date_end-date_start))
echo "096_cpus $time_run seconds"

rm -f GTH_BASIS_SETS POTENTIAL *xyz *restart          #remove useless big files (please!) 

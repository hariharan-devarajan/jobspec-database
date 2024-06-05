#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH -o %x-%N-%j.out
#SBATCH -e %x-%N-%j.err

source /etc/profile.d/modules.sh
module load rocm/5.2.3

tmp=/tmp/$USER/tmp-$$
mkdir -p $tmp

singularity run /shared/apps/bin/specfem3d_globe1ee10977-20210321.sif /bin/bash -c "cp -r /opt/specfem3d_globe ./"
singularity run --bind ./specfem3d_globe:/opt/specfem3d_globe /shared/apps/bin/specfem3d_globe1ee10977-20210321.sif /bin/bash -c "benchmark global_s362ani_shakemovie -o /tmp/out"
singularity run --bind ./specfem3d_globe:/opt/specfem3d_globe /shared/apps/bin/specfem3d_globe1ee10977-20210321.sif /bin/bash -c "cd /opt/specfem3d_globe/EXAMPLES/regional_Greece_small; ./run_this_example.sh"

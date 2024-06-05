#!/bin/bash
#SBATCH --job-name=dask_job
#SBATCH -n 102
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3G
#SBATCH --time=02:00:00
#SBATCH --partition=cluster

# make sure we have Singularity
module load singularity/3.5.2

CONTAINER_FILE="pangeo-notebook_2022.07.27.sif"

# set up jupyterlab
srun --ntasks=1 -N1 --exclusive singularity run -B /sfs -B /gxfs_work1 -B $PWD:/work --pwd /work $CONTAINER_FILE bash -c \
        ". /srv/conda/etc/profile.d/conda.sh && conda activate base \
        && jupyter-lab --ip='0.0.0.0'" &

# Set up dask cluster
srun --ntasks=1 -N1 --exclusive singularity run -B /sfs -B /gxfs_work1 -B $PWD:/work --pwd /work $CONTAINER_FILE bash -c \
        ". /srv/conda/etc/profile.d/conda.sh && conda activate base \
        && dask-scheduler --scheduler-file scheduler.json" &
sleep 15  # allow for the scheduler to come up
srun --ntasks=100 --exclusive singularity run -B /sfs -B /gxfs_work1 -B $PWD:/work --pwd /work $CONTAINER_FILE bash -c \
        ". /srv/conda/etc/profile.d/conda.sh && conda activate base \
        && dask-worker --scheduler-file scheduler.json --nthreads=10" &

wait

# print resource infos
jobinfo

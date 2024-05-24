#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=168:00:00
#SBATCH --requeue
#SBATCH --account=def-adagher

# Load necessary modules
module load StdEnv/2020 intel/2020.1.217 mrtrix/3.0.1 nextflow apptainer

export NXF_CLUSTER_SEED=$(shuf -i 0-16777216 -n 1)

# Run TractoFlow with all the necessary parameters
nextflow run /home/mtweed/scratch/tractoflow/main.nf --input /home/mtweed/scratch/tractoflow_hcp_dwi/hcp/ -profile skip_preprocessing --dti_shells "0 1000 2000 3000" --fodf_shells "0 1000 2000 3000" --sh_order "8" -with-singularity /home/mtweed/scratch/tractoflow/scilus_1.5.0.sif -with-mpi -resume

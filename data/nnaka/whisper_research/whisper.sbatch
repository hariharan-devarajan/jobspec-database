#!/bin/bash 
#SBATCH --nodes=3                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=60GB
#SBATCH --gres=gpu
# #SBATCH --job-name=whisper-research
# #SBATCH --output=whisper-research.out

module purge

singularity exec --nv \
	    --overlay /scratch/nn1331/whisper/whisper.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python whisper.py"

#!/bin/bash
#SBATCH -p carlsonlab-gpu --account=carlsonlab --gres=gpu:1 --mem=64G
#SBATCH --job-name=img_prep
#SBATCH --output=img_prep.out
#SBATCH --error=img_prep.err
#SBATCH -c 2
#SBATCH --nice

srun singularity exec --nv --bind /work/ld243 /datacommons/carlsonlab/Containers/multimodal_gp.simg python Image_Preprocessing_1.py
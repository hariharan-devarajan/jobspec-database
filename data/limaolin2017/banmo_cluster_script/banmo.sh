#!/bin/bash
#SBATCH -J 3D_reconstruction      # name
#SBATCH -p high                      # Queue name
#SBATCH -N 2                   # total number of nodes requested (16 cores/node)
#SBATCH --gres=gpu:2                         # GPUs per node
#SBATCH --mem=16g                     # Memory per node.
#SBATCH --exclude=node[001-018],node[031-032]  # Exclude the nodes with faulty GPUs
#SBATCH --time=99:99:99                # maximum execution time (HH:MM:SS)
#SBATCH -o slurm_3D.%N.%J.out       # Name of the standard output file
#SBATCH -e slurm_3D.%N.%J.err        # Name of the standard error file
#SBATCH --distribution=block

# Load the required modules (adjust for your cluster environment)

module load Miniconda3

cd ./banmo
eval "$(conda shell.bash hook)"

# Activate your conda environment
source /soft/easybuild/x86_64/software/Miniconda3/4.9.2/etc/profile.d/conda.sh
conda activate banmo-cu113

export LD_LIBRARY_PATH=/gpfs/home/mli/.conda/envs/banmo-cu113/lib/:$LD_LIBRARY_PATH


seqname="cat-pikachiu"

python preprocess/img2lines.py --seqname $seqname

bash scripts/template.sh 0,1 $seqname 10001 "no" "no"

bash scripts/render_mgpu.sh 0,1 $seqname logdir/$seqname-e120-b256-ft2/params_latest.pth \
        "0 1 2 3 4 5 6 7 8 9 10" 256
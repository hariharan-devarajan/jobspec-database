#!/bin/bash
#SBATCH --mail-user=rf2485@nyulangone.org
#SBATCH --mail-type=ALL
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --array=1-322
#SBATCH -o ./slurm_output/fmap_corr/slurm-%A_%a.out

module load miniconda3/gpu/4.9.2
# conda create -n nipype
conda activate nipype
# conda install --channel conda-forge nipype
# conda install -c mrtrix3 mrtrix3
# pip install nipype
module load fsl/.6.0.6
export LD_LIBRARY_PATH=/lib
module load ants/2.1.0
module load freesurfer/7.4.1

subj_list=$(cut -f1 subjectsfile.txt)
subj_list=($subj_list)
subj_num=$(($SLURM_ARRAY_TASK_ID-1))
subj=${subj_list[$subj_num]}

python3 fmap_corr.py $subj

#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p zhuang,shared
#SBATCH -t 7-00:00:00
#SBATCH --mem 4000
#SBATCH --open-mode=append
#SBATCH -o /n/home06/lsepulvedaduran/Software/merfish-parameters/slurm/out/BC071_sample_03_seg_cv2.out
#SBATCH -e /n/home06/lsepulvedaduran/Software/merfish-parameters/slurm/err/BC071_sample_03_seg_cv2.err

date +'Starting at %R.'

source centos7-modules.sh
module load Anaconda3/5.0.1-fasrc01
source activate merlin_env
module load gcc/8.2.0-fasrc01
module load fftw
which python
echo BC071_sample_03

merlin -k snakemake_parameters.json \
       -a merlin_analysis_BC071_seg_cv2.json \
       -o data_organization_BC071_3.csv \
       -p positions_BC071_sample_03.txt \
       -c C1E1_codebook.csv \
       -m MERFISH3.json \
       -n 1000 \
       191212_BC071_MERFISH/sample_03

date +'Finished at %R.'

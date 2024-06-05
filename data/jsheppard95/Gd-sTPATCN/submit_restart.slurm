#!/bin/bash -l
#SBATCH -N 1 --partition=gpu --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH -t 36:00:00  # Run time (hh:mm:ss)
#SBATCH --mail-user=sheppard@ucsb.edu
#SBATCH --mail-type=all
#SBATCH --job-name=Gd-sTPATCN_2V1A_FMN_05NPT_100-150ns_Restart
#SBATCH -o ./2V1A_FMN/05NPT_100-150ns/output.%j.out
##  will use the GPU partition

cd $SLURM_SUBMIT_DIR

module load singularity 

/bin/hostname
singularity run --nv  /sw/singularity/SingularityImages/gromacs-2021.sif << EOF
## Put your commands of what to do in the gromacs container  in here
#
cd /home/jsheppard/research/Gd-sTPATCN/2V1A_FMN/05NPT_100-150ns/
echo "getting ready to run"
gmx mdrun  -ntmpi 1 -ntomp $SLURM_NTASKS  -nb gpu  -pin on -deffnm npt  -cpi npt.cpt
date
##  End of Gromacs job commands
EOF
#  just to pause it 10 seconds...
sleep 10

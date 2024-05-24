#!/bin/bash
#SBATCH --account cin_staff
#SBATCH --partition prod
##SBATCH --qos=leo_qos_dbg
#SBATCH --time 00:20:00     # format: HH:MM:SS
#SBATCH --nodes 1                # 1 node
#SBATCH --ntasks-per-node=1 # 8 tasks out of 128
##SBATCH --gres=gpu:1        # 1 gpus per node out of 4
#SBATCH --job-name=PLUTO
#SBATCH --mail-type=NONE
#SBATCH --mail-user=m.mencagli@cineca.it

## load needed modules on the node
source ${HOME}/modules_files/pluto_mod

## print GPUs node information
nvidia-smi

## run the pluto tests
cd gpluto_cpp/
python3 test_pluto.py --auto-update
python3 test_pluto.py --start-from 0:0


 #nsys profile --trace=cuda,mpi,nvtx,openacc --force-overwrite true -o report_MHD_highorder_BLAST_B1_gpu4_leo mpirun -np 4 ./pluto     #profile multiple gpus
 #nsys profile --force-overwrite true -o report_MHD_BLAST_B3_304cube_leo_o1 ./pluto -maxsteps 10 -xres 304                             #profile single gpu
 #ncu -k regex:Roe_Solver -f -o NCUreport_MHD_BLAST_B3_304cube_leo ./pluto -maxsteps 2 -xres 304                                       #profile a kernel that include "Roe_Solver" in its name

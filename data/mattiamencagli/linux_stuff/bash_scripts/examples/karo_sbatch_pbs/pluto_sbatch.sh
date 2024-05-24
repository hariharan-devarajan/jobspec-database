#!/bin/bash
##################SBATCH --account DD-23-44
#SBATCH --account FTA-23-25
#SBATCH --partition qcpu
#SBATCH --time 00:20:00            # format: HH:MM:SS
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=128       # n tasks out of 32
#######SBATCH --cpus-per-task=1    # 4 virtual cpu per task. So I use n physical core
#######SBATCH --gres=gpu:4               # 1 gpus per node out of 4
#SBATCH --job-name=PLUTO
#SBATCH --mail-type=NONE
#SBATCH --mail-user=m.mencagli@cineca.it

## load needed modules on the node
source ${HOME}/modules_files/pluto_mod

## print GPUs node information
#nvidia-smi

## run the pluto tests
#cd gpluto_cpp/
#python3 test_pluto.py --auto-update
#time python3 test_pluto.py --start-from 0:0

#cd ${HOME}/programming/gpluto_cpp/Test_Problems/MHD/Blast
#pluto_select 3  #build number + mpi
#time make -j -B > makeout_prova 2>&1
#nsys profile --trace=cuda,mpi,nvtx,openacc --force-overwrite true -o report_MHD_highorder_BLAST_B1_gpu4_leo mpirun -np 4 ./pluto
#nsys profile --force-overwrite true -o report_MHD_BLAST_B3_304cube_leo_o2 ./pluto -maxsteps 10 -xres 304
#ncu --target-processes all -k regex:Particles_LP_UpdateSpectra_325 --set=full --import-source=yes --launch-count 1 --launch-skip 100 -f -o NCUreport_MM_leo ./pluto -xres 304

cd ${HOME}/programming/gpluto_from_leo/Test_Problems/MHD/Orszag_Tang
#time make -j -B > makeout_prova 2>&1
#nsys profile --trace=cuda,mpi,nvtx,openacc --force-overwrite true -o report_nsys_MM_leo ./pluto
#ncu --target-processes all -k regex:Particles_LP_UpdateSpectra_323 --set=full --import-source=yes --launch-count 1 --launch-skip 194 -f true -o NCUreport_MM2_leo "./pluto" -no-write
#ncu --kernel-name Particles_LP_UpdateSpectra_323 --set=full --import-source=yes --launch-skip 194 --launch-count 1 "./pluto" -no-write

#./pluto

#mpirun -np 128 ./pluto -maxsteps 60


#for NN in 1 2 4 8 16 32 64 128 256; do            
for NN in 1; do            
    NG=$(( ${NN} * 128 ))                          
    mpirun -np ${NG} ./pluto -maxsteps 60         
    cp -v pluto.0.log plutolog_KAROcpu_${NN}          
done 









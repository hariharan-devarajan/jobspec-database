#!/bin/bash
#SBATCH --account cin_staff
#SBATCH --partition boost_usr_prod
#######SBATCH --qos boost_qos_bprod
#SBATCH --time 1:00:00            # format: HH:MM:SS
#SBATCH --nodes 32
#SBATCH --ntasks-per-node=32        # n tasks out of 32
#SBATCH --cpus-per-task=1    # 4 virtual cpu per task. So I use n physical core
#SBATCH --gres=gpu:4               # 1 gpus per node out of 4
#SBATCH --job-name=PLUTO_sW
#SBATCH --mail-type=NONE
#SBATCH --mail-user=m.mencagli@cineca.it

## load needed modules on the node
source ${HOME}/modules_files/pluto_mod 23.11

#module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8

## print GPUs node information
#nvidia-smi

cd ${HOME}/programming/gpluto_cpp/Test_Problems/MHD/Orszag_Tang
pwd

make -j -B > makeout 2>&1

#for NN in 1 2 4 8 16 32 64 128 256; do 
for NN in 32; do 
	NG=$(( ${NN} * 32 ))
	cp -v pluto_${NN}.ini pluto.ini
	mpirun -np ${NG} ./pluto -maxsteps 40
	cp -v pluto.0.log /leonardo_scratch/large/userinternal/mmencag1/WEAK_test_GPUs/plutologWEAK_cpu_${NN}
done


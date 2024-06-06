#!/bin/bash
#PBS -l nodes=1:ppn=7:gpus=1:exclusive_process
#PBS -l walltime=72:00:00
#PBS -N aqOTL

unset CUDA_VISIBLE_DEVICES

module purge
module load chem/gromacs/5.1.4-gnu-4.9
module load devel/cuda/7.5 numlib/openblas/0.2.18-gnu-4.9

cd $PBS_O_WORKDIR

# Set some environment variables 
FREE_ENERGY=$PBS_O_WORKDIR
echo "Free energy home directory set to $FREE_ENERGY"

MDP=$FREE_ENERGY/mdp_files
echo ".mdp files are stored in $MDP"

LAMBDA=0

# A new directory will be created for each value of lambda and
# at each step in the workflow for maximum organization.

mkdir Lambda_$LAMBDA
cd Lambda_$LAMBDA

#################################
# ENERGY MINIMIZATION 1: STEEP  #
#################################
echo "Starting minimization for lambda = $LAMBDA..." 

mkdir EM_1 
cd EM_1

# Iterative calls to grompp and mdrun to run the simulations

gmx grompp -f $MDP/EM/em_steep_$LAMBDA.mdp -c $FREE_ENERGY/../coord/start.gro -p $FREE_ENERGY/../topo/topo.top -o min_$LAMBDA.tpr -maxwarn 33

mdrun_s_gpu -ntomp $PBS_NP -deffnm min$LAMBDA

sleep 10

#####################
# NVT EQUILIBRATION #
#####################
echo "Starting constant volume equilibration..."

cd ../
mkdir NVT
cd NVT

gmx grompp -f $MDP/NVT/nvt_$LAMBDA.mdp -c ../EM_1/min$LAMBDA.gro -p $FREE_ENERGY/../topo/topo.top -o nvt$LAMBDA.tpr -maxwarn 33

mdrun_s_gpu -ntomp $PBS_NP -deffnm nvt$LAMBDA

echo "Constant volume equilibration complete."

sleep 10

#####################
# NPT EQUILIBRATION #
#####################
echo "Starting constant pressure equilibration..."

cd ../
mkdir NPT
cd NPT

gmx grompp -f $MDP/NPT/npt_$LAMBDA.mdp -c ../NVT/nvt$LAMBDA.gro -p $FREE_ENERGY/../topo/topo.top -t ../NVT/nvt$LAMBDA.cpt -o npt$LAMBDA.tpr -maxwarn 33

mdrun_s_gpu -ntomp $PBS_NP -deffnm npt$LAMBDA

echo "Constant pressure equilibration complete."

sleep 10

#################
# PRODUCTION MD #
#################
echo "Starting production MD simulation..."

cd ../
mkdir Production_MD
cd Production_MD

gmx grompp -f $MDP/Production_MD/md_$LAMBDA.mdp -c ../NPT/npt$LAMBDA.gro -p $FREE_ENERGY/../topo/topo.top -t ../NPT/npt$LAMBDA.cpt -o md$LAMBDA.tpr -maxwarn 33

mdrun_s_gpu -ntomp $PBS_NP -deffnm md$LAMBDA

echo "Production MD complete."

# End
echo "Ending. Job completed for lambda = $LAMBDA"

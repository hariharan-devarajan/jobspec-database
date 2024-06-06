#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=ku_00004 -A ku_00004
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N pcangsd
### Output files (comment out the next 2 lines to get the job name used instead)
##PBS -e pcanagsd_test.err
##PBS -o pcangsd.log
### Only send mail when job is aborted or terminates abnormally
#PBS -m n
### Number of nodes, request 196 cores from 7 nodes
#PBS -l nodes=1:ppn=40:thinnode
### Requesting time - 720 hours
#PBS -l walltime=24:00:00
#PBS -l mem=188gb
 
### Here follows the user commands:
# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
# NPROCS will be set to 196, not sure if it used here for anything.
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS nodes
  
module load tools anaconda2/4.4.0 pcangsd/0.97 
 
export OMP_NUM_THREADS=60
# Using 192 cores for MPI threads leaving 4 cores for overhead, '--mca btl_tcp_if_include ib0' forces InfiniBand interconnect for improved latency
#mpirun -np 40 $mdrun -s gmx5_double.tpr -plumed plumed2_path_re.dat -deffnm md-DTU -dlb yes -cpi md-DTU -append --mca btl_tcp_if_include ib0


folder=$(echo $(pwd) | sed -e 's/.*folder\(.*\)_out.*/\1/')

gzip new_file"$folder".txt
python /home/projects/ku_00004/apps/pcangsd/pcangsd.py -beagle new_file"$folder".txt.gz -o pca_486  -iter 100 
#-admix -admix_iter 100 -selection -inbreed 1 -kinship
#combined_beagle_pruned_gl.txt.gz

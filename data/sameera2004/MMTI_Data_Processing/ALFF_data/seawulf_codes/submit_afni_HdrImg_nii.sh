#!/bin/bash 

#set the number of nodes and processes per node and wallclock time
#PBS -l nodes=1:ppn=28,walltime=04:00:00

#PBS -q short

# set name of job
#PBS -N hdr_img_nii

# mail alert at (b)eginning, (e)nd and (a)bortion of execution
#PBS -m bea

# send mail to the following address
#PBS -M sameera.abeykoon@stonybrookmedicine.edu

# use submission environment 
#PBS -V

# start job from the directory it was submitted (see following)
cd $PBS_O_WORKDIR
# cd /gpfs/projects/MoellerGroup/TASK_scripts 

module load shared
module load gsl/2.4.0
module load afni/17.2.05
module load matlab/2017a

#StudyFolder="/gpfs/projects/VanSnellenbergGroup/PET_images"
#niiFolder="/gpfs/projects/VanSnellenbergGroup/jxvs01_PET_images"

sh /gpfs/projects/VanSnellenbergGroup/S_codes/afni_3dcalc.sh

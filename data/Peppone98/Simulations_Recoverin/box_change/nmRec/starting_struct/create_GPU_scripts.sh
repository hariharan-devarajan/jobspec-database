#!/bin/bash

n_cores=1
n_mpiprocs=1
n_walk=(0 1 2 3 4 5 6 7)

len=${#n_walk[@]}


# Create the folders
for (( i=0; i<$len; i++ ))
do
  mkdir MD/W${n_walk[i]}
  touch MD/W${n_walk[i]}/md_${n_walk[i]}.pbs
done


# pbs files
for (( i=0; i<$len; i++ ))
do
  echo "#!/bin/bash\n#PBS -l select=1:ncpus=${n_cores}:mpiprocs=${n_mpiprocs}:ngpus=1:mem=1GB\n#PBS -l walltime=01:00:00
  \n#PBS -q short_gpuQ\n#PBS -N md_${n_walk[i]}\n#PBS -o md_${n_walk[i]}_out\n#PBS -e md_${n_walk[i]}_err
  \n\nmodule load gcc91\nmodule load openmpi-3.0.0\nmodule load BLAS\nmodule load gsl-2.5\nmodule load lapack-3.7.0
  \nmodule load cuda-11.3\n" > MD/W${n_walk[i]}/md_${n_walk[i]}.pbs

  awk 'NR==18' template.pbs >> MD/W${n_walk[i]}/md_${n_walk[i]}.pbs
  awk 'NR==20' template.pbs >> MD/W${n_walk[i]}/md_${n_walk[i]}.pbs
  awk 'NR==22' template.pbs >> MD/W${n_walk[i]}/md_${n_walk[i]}.pbs

  echo "\nexport OMP_NUM_THREADS=1
  \n/apps/openmpi-3.0.0/bin/mpirun -np ${n_mpiprocs} /home/giuseppe.gambini/usr/installations/gromacs/bin/gmx_mpi mdrun -s md_${n_walk[i]}.tpr -nb gpu -pme auto" >> MD/W${n_walk[i]}/md_${n_walk[i]}.pbs
done


# Create the .tpr files inside each subfolder
for (( i=0; i<$len; i++ ))
do
  mv md_${n_walk[i]}.tpr MD/W${n_walk[i]}
done
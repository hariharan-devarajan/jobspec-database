#!/bin/bash
#SBATCH -p debug 
#SBATCH -N 1 
#SBATCH -t 00:05:00
#SBATCH -J h5write
#SBATCH -e %j.err
#SBATCH -o %j.out
#SBATCH -C haswell
##arguments: 
## -i: inputfilename, -o:outputfilename, -b: collective_buffersize, -n: collective_buffernodes, -k: iscollective, -v: datasetname
  
nodes=1        #Nodes 32
nprocs=2      #Processes 1024
#cbn=1          #Aggregators 32
#cbs=16777216   #Collective Buffer size 16MB
iscollective=1 #Collective IO
dimx=320      #Size of X dimension
dimy=3000      #Size of Y dimension
lost=2         #OST 72
hostpartion=haswell
#filename_c=$SCRATCH/hdf-data/ost${lost}/test_${hostpartion}_${nprocs}_${SLURM_JOBID}_c.h5
filename_py=$SCRATCH/hdf-data/ost${lost}/test_${hostpartion}_${nprocs}_${SLURM_JOBID}_py.h5
rm -rf $SCRATCH/hdf-data/ost${lost}/
mkdir -p $SCRATCH/hdf-data/ost${lost}/
lfs setstripe -c $lost $SCRATCH/hdf-data/ost${lost}/
#cmd="srun -n $nprocs  ./h5write -f $filename -b $cbs -n $cbn -k $iscollective -x $dimx -y $dimy"

#echo $hostpartion
#printf "hdf5:\n"
#export  MPICH_MPIIO_HINTS_DISPLAY=1
#module load cray-hdf5-parallel/1.8.16
module load darshan/3.0.1
#cc -o h5write h5write.c
#sbcast --compress=lz4 ./h5write /tmp/h5write
#cmd="srun -n $nprocs  ./h5write -k $iscollective -f $filename_c -x $dimx -y $dimy"
#echo $cmd
#time $cmd
#sleep 10

printf "h5py:\n"
module load python/2.7-anaconda
module load h5py-parallel
export LD_PRELOAD=/opt/cray/pe/mpt/7.4.4/gni/mpich-gnu/5.1/lib/libmpich.so:/usr/common/software/darshan/3.0.1/lib/libdarshan.so:$LD_PRELOAD
#sbcast --compress=lz4 ./h5write.py /tmp/h5write.py
cmd="srun -n $nprocs python-mpi h5write.py $iscollective $filename_py $dimx $dimy"
echo $cmd
time $cmd
#printf "\n"
#ls -alh $filename_c
#printf "\n"
#ls -alh $filename_py
#module list
#printf "\n"
#lfs getstripe $filename_c
#printf "\n"
#lfs getstripe $filename_py
#rm $filename

#!/bin/sh
# Script for running a simulation job in the SLURM system
#
# Written by Shyam Kumar Sudhakar, Ivan Raikov, Tom Close, Rodrigo Publio, Daqing Guo, and Sungho Hong
# Computational Neuroscience Unit, Okinawa Institute of Science and Technology, Japan
# Supervisor: Erik De Schutter
#
# Correspondence: Sungho Hong (shhong@oist.jp)
#
# September 16, 2017

## Some parameters for running a SLURM job
#SBATCH --job-name=GL_SIM
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=10G
#SBATCH --time=120:30:00
#SBATCH --ntasks=120
#SBATCH --cpus-per-task=2
#SBATCH --input=none
## Standard output and standard error files
#SBATCH --output=SHAREDDIR/simulation.out.log
#SBATCH --error=SHAREDDIR/simulation.err.log

export PATH=... # Set paths for python, etc. here
NEURONHOME=... # Set your NEURONHOME here
export PATH=$NEURONHOME/nrn/x86_64/bin:$NEURONHOME/iv/x86_64/bin:$PATH
export LD_LIBRARY_PATH=$NEURONHOME/nrn/x86_64/lib:$NEURONHOME/iv/x86_64/lib:$LD_LIBRARY_PATH

echo PYTHONPATH is $PYTHONPATH

echo "==============Starting mpirun==============="
cd SHAREDDIR/model

mpirun nrniv -mpi -python main.py

echo "==============Mpirun has ended==============="

## Copy all the output data
mkdir $HOME/work/output.$JOB_ID
cp -v *.dat $HOME/work/output.$JOB_ID
cp -v *.bin $HOME/work/output.$JOB_ID
cp -R $PARAMDIR $HOME/work/output.$JOB_ID

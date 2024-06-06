#!bin/bash
#BSUB -q hpc
#BSUB -J HPCPYTHON[2,29,71,73,127] 
#BSUB -n 1
#BSUB -W 3 
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u s204123@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o %J_%I.out
#BSUB -e %J_%.out

echo "hi job" $LSB_JOBINDEX

#Reduction full

#!/bin/sh

### General options
### –- specify queue --
#BSUB -q p1
### -- set the job Name --
#BSUB -J eval_bloom7b_instruct
### -- ask for number of cores (default: 1) --
#BSUB -n 16
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 3:00
# request 5GB of system-memory
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5000MB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u caap@itu.dk
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /dtu/p1/johlau/logs/R-eval_bloom7b_instruct_%J.out
#BSUB -e /dtu/p1/johlau/logs/R-eval_bloom7b_instruct_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/12.1

bash /dtu/p1/johlau/Tk-Instruct/scripts/run_bloom7b.sh
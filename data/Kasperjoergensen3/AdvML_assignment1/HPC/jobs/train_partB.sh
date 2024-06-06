### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J PartB
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o HPC/outputs/PartB_%J.out
#BSUB -e HPC/outputs/PartB_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.6

# Your Conda initialization step (path might be different based on your installation)
source /zhome/4b/e/155339/miniconda3/etc/profile.d/conda.sh
#activate conda environment
conda activate Denoising_EEG

# WANDB_API_KEY=${WANDB_API_KEY}
# # Log in to WandB
# wandb login $WANDB_API_KEY
#run the training script
#make train_gaussian_gpu

make run_partB

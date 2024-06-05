#!/bin/bash
#SBATCH --job-name=nn_art_transfer        # Job name
#SBATCH --mail-type=ALL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ap1751@york.ac.uk      # Where to send mail  
#SBATCH --ntasks=1                       # Run on a single CPU
#SBATCH --mem=4gb                        # Job memory request
#SBATCH --time=00:30:00                  # Time limit hrs:min:sec
#SBATCH --output=nn_job_%j.log        # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
echo "Pulling latest"
git pull
echo "Loading Modules"
module load lang/Python/3.8.2-GCCcore-9.3.0
module load system/CUDA/10.1.243-GCC-8.3.0
module load numlib/cuDNN/7.6.4.38-gcccuda-2019b
nvidia-smi --gpu-reset
if [ -d "venv" ]; then
   echo "Found previous venv deleting"
   rm -rf venv
fi
if [ -d "generated_images" ]; then
    echo "Found previous generated images deleting"
    rm -rf generated_images
fi
mkdir -p generated_images
echo "Setting up venv"
pwd
python3 -m venv venv
source venv/bin/activate
echo "upgarding pip"
python3 -m pip install --upgrade pip
echo "Installing requiremnts"
python3 -m pip install -r requirements.txt
python3 index.py

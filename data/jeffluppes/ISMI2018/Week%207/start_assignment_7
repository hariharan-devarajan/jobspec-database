#!/bin/bash
#SBATCH -t 3:00:00
`#SBATCH --reservation=rumc_lab`
#SBATCH -N 1
#SBATCH -p gpu
module load eb
module load Python/3.6.3-foss-2017b
python3 -m pip install jupyter --user
export PATH=~/.local/bin:$PATH
pip install --upgrade pip --user
pip install opencv-python --user
pip install imgaug --user
pip install tensorflow-gpu==1.4.1 --user
pip install keras --user
pip install tqdm --user
pip install jupyter --user
pip install h5py --user
pip install requests --user
module load cuda/8.0.44
module load cudnn/8.0-v6.0
module load gcc/4.9.2

ssh -o StrictHostKeyChecking=no -f -N -p 22 -R 5255:localhost:5255 int3
#$SLURM_SUBMIT_HOST
jupyter notebook --no-browser --port 5255 --NotebookApp.token=''

#!/usr/local_rwth/bin/zsh


### Job name
#SBATCH --job-name=FredericMasterThesisGAN

### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output=output_%J.txt

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=5

### Request the amount of memory you need for your job.
### You can specify this in either MB (1024M) or GB (4G).
#SBATCH --mem-per-cpu=2

### Request a host with a Volta GPU
### If you need two GPUs, change the number accordingly
#SBATCH --gres=gpu:volta:1

### if needed: switch to your working directory (where you saved your program)
#cd $HOME/Master-Thesis_GAN-level-gen/src/

### Load modules
module load python/3.8.7
module load cuda/11.0
module load cudnn/8.0.5

pip3 install --user tensorflow

### Make sure you have 'tensorflow-gpu' installed, because using
### 'tensorflow' will lead to your program not using the requested
### GPU.

### Execute your application
python3 trainer/TrainNeuralNetwork.py